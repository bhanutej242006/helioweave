"""Small, fully trainable NumPy LSTM with BPTT and Adam (no fixed random gates)."""
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


class LSTM:
    def __init__(self, inputs, hidden, outputs, seed=42):
        self.inputs, self.hidden, self.outputs = inputs, hidden, outputs
        rng = np.random.default_rng(seed)
        self.p = {'W': rng.normal(0, 1/np.sqrt(inputs+hidden), (inputs+hidden, 4*hidden)),
                  'b': np.zeros(4*hidden), 'V': rng.normal(0, .1, (hidden, outputs)),
                  'a': np.zeros(outputs)}
        self.p['b'][hidden:2*hidden] = 1  # forget bias

    def forward(self, X, cache=False):
        n, _, _ = X.shape
        h = np.zeros((n, self.hidden)); c = h.copy(); tape = []
        for t in range(X.shape[1]):
            z = np.concatenate([X[:, t], h], axis=1)
            gates = z @ self.p['W'] + self.p['b']
            i, f, o, g = np.split(gates, 4, axis=1)
            i, f, o, g = sigmoid(i), sigmoid(f), sigmoid(o), np.tanh(g)
            cp = c; c = f*c + i*g; h = o*np.tanh(c)
            if cache: tape.append((z, i, f, o, g, cp, c))
        y = h @ self.p['V'] + self.p['a']
        return (y, h, tape) if cache else y

    def gradients(self, X, Y):
        y, h, tape = self.forward(X, True)
        dy = 2*(y-Y)/Y.size
        grad = {k: np.zeros_like(v) for k, v in self.p.items()}
        grad['V'] = h.T@dy; grad['a'] = dy.sum(0)
        dh = dy@self.p['V'].T; dc = np.zeros_like(dh)
        for z, i, f, o, g, cp, c in reversed(tape):
            tc = np.tanh(c); dc = dc + dh*o*(1-tc*tc)
            dgates = np.concatenate([dc*g*i*(1-i), dc*cp*f*(1-f),
                                     dh*tc*o*(1-o), dc*i*(1-g*g)], axis=1)
            grad['W'] += z.T@dgates; grad['b'] += dgates.sum(0)
            dh = (dgates@self.p['W'].T)[:, self.inputs:]; dc = dc*f
        return float(np.mean((y-Y)**2)), grad

    def fit(self, X, Y, VX, VY, epochs=20, batch=64, lr=.003):
        rng = np.random.default_rng(42)
        m = {k: np.zeros_like(v) for k,v in self.p.items()}; v = {k: a.copy() for k,a in m.items()}
        step = 0; best = float('inf'); checkpoint = None; history = []
        for epoch in range(epochs):
            order = rng.permutation(len(X)); losses=[]
            for start in range(0,len(X),batch):
                ids=order[start:start+batch]; loss, grad=self.gradients(X[ids],Y[ids]); losses.append(loss)
                norm=np.sqrt(sum(np.sum(g*g) for g in grad.values())); step+=1
                for k in self.p:
                    g=grad[k]*min(1, 1/(norm+1e-12))
                    m[k]=.9*m[k]+.1*g; v[k]=.999*v[k]+.001*g*g
                    self.p[k]-=lr*(m[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
            val=float(np.mean((self.forward(VX)-VY)**2))
            history.append({'epoch':epoch+1,'train_mse':float(np.mean(losses)),'validation_mse':val})
            if val<best: best=val; checkpoint={k:a.copy() for k,a in self.p.items()}
        self.p=checkpoint
        return history

    def save(self,path):
        np.savez(path, **self.p, dims=[self.inputs,self.hidden,self.outputs])

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            net=cls(*map(int,data['dims'])); net.p={k:data[k].copy() for k in net.p}
        return net
