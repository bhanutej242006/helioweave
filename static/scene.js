import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';

export class Campus {
 constructor(host){
  this.host=host;this.paused=matchMedia('(prefers-reduced-motion: reduce)').matches;this.clock=0;this.flows=[];this.tags=[];this.sunValues={zenith:45,azimuth:160};this.ratio=.6;this.evRatio=.35;this.gridRatio=.15;
  try{this.renderer=new THREE.WebGLRenderer({antialias:true,alpha:true,powerPreference:'high-performance'});}catch(e){document.getElementById('scene-fallback').hidden=false;this.failed=true;return;}
  const r=this.renderer;r.setPixelRatio(Math.min(devicePixelRatio,1.6));r.shadowMap.enabled=true;r.shadowMap.type=THREE.PCFSoftShadowMap;r.outputColorSpace=THREE.SRGBColorSpace;r.toneMapping=THREE.ACESFilmicToneMapping;r.toneMappingExposure=1.2;host.prepend(r.domElement);
  this.scene=new THREE.Scene();this.scene.fog=new THREE.FogExp2(0x14252f,.009);this.camera=new THREE.PerspectiveCamera(38,1,.1,150);this.camera.position.set(18,20,24);this.controls=new OrbitControls(this.camera,r.domElement);this.controls.target.set(0,.1,0);this.controls.enableDamping=true;this.controls.dampingFactor=.07;this.controls.minDistance=17;this.controls.maxDistance=52;this.controls.maxPolarAngle=Math.PI*.46;this.controls.minPolarAngle=.3;this.controls.enablePan=false;
  this.scene.add(new THREE.HemisphereLight(0xc4e4f2,0x14212b,2.6));this.sunlight=new THREE.DirectionalLight(0xffdfb6,3.2);this.sunlight.position.set(-10,19,8);this.sunlight.castShadow=true;this.sunlight.shadow.mapSize.set(2048,2048);Object.assign(this.sunlight.shadow.camera,{left:-19,right:19,top:19,bottom:-19,near:1,far:65});this.sunlight.shadow.normalBias=.05;this.scene.add(this.sunlight);
  const rim=new THREE.DirectionalLight(0x66cce1,2);rim.position.set(6,5,-12);this.scene.add(rim);
  this.build();new ResizeObserver(()=>this.resize()).observe(host);this.resize();this.renderer.setAnimationLoop(t=>this.draw(t));
 }
 material(color,metal=.1,rough=.6){return new THREE.MeshStandardMaterial({color,metalness:metal,roughness:rough});}
 mesh(geometry,material,x=0,y=0,z=0,parent=this.scene){const m=new THREE.Mesh(geometry,material);m.position.set(x,y,z);m.castShadow=true;m.receiveShadow=true;parent.add(m);return m;}
 box(w,h,d,mat,x,y,z,parent=this.scene){return this.mesh(new THREE.BoxGeometry(w,h,d),mat,x,y,z,parent);}
 tube(points,color,r=.035,parent=this.scene){const curve=new THREE.CatmullRomCurve3(points.map(p=>new THREE.Vector3(...p)));const m=this.mesh(new THREE.TubeGeometry(curve,48,r,6,false),new THREE.MeshStandardMaterial({color,emissive:color,emissiveIntensity:.35,roughness:.4}),0,0,0,parent);return curve;}
 label(text,pos){const el=document.createElement('div');el.className='scene-label';el.textContent=text;document.getElementById('scene-labels').append(el);this.tags.push({el,p:new THREE.Vector3(...pos)});}
 build(){
  const dark=this.material(0x1b2c37,.25),edge=this.material(0x415761,.65,.35),concrete=this.material(0x65757a,.1,.9),asphalt=this.material(0x172631,.05,.9),white=this.material(0xa6b5ba,.1,.55),steel=this.material(0x8fabb4,.8,.3);
  this.box(24,.65,18,dark,0,-.65,0);this.box(23.8,.08,17.8,edge,0,-.29,0);this.box(23.5,.12,17.5,asphalt,0,-.2,0);
  const ground=this.mesh(new THREE.PlaneGeometry(150,150),this.material(0x14222c,0,1),0,-1.04,0);ground.rotation.x=-Math.PI/2;
  const grid=new THREE.GridHelper(100,60,0x304854,0x213542);grid.position.y=-1;grid.material.transparent=true;grid.material.opacity=.4;this.scene.add(grid);
  // Thin architectural lane marks and service road.
  this.box(21,.018,2.15,this.material(0x20333f),0,-.12,2.1);
  for(let x=-10;x<11;x+=1.8)this.box(.7,.022,.045,white,x,-.09,2.1);
  for(let x=-10;x<=10;x+=5)this.box(.045,.02,15,edge,x,-.12,0);
  // Solar array: two banks, individual framed modules and visible cell segmentation.
  this.panels=[];const cell=new THREE.MeshPhysicalMaterial({color:0x285a82,metalness:.3,roughness:.4,clearcoat:1,clearcoatRoughness:.2});
  const cellLine=this.material(0x577991,.5,.4);
  for(let bank=0;bank<2;bank++)for(let row=0;row<3;row++)for(let col=0;col<4;col++){
    const x=-8.6+col*1.8+bank*8.1,z=-5.9+row*2.05;
    const group=new THREE.Group();group.position.set(x,1.05,z);group.rotation.x=.32;this.scene.add(group);
    this.box(1.66,.09,1.67,steel,0,0,0,group);const surface=this.box(1.55,.028,1.56,cell,0,.059,0,group);this.panels.push(surface);
    for(let j=-2;j<=2;j++)this.box(.008,.006,1.5,cellLine,j*.255,.078,0,group);
    for(let j=-2;j<=2;j++)this.box(1.5,.006,.008,cellLine,0,.078,j*.255,group);
    this.box(.07,1.1,.07,edge,x-.65,.38,z);this.box(.07,1.1,.07,edge,x+.65,.38,z);
  }
  // Power conversion skid: slatted inverter enclosure, transformer and bus.
  this.box(3.25,.16,2.35,concrete,8.7,0,-4.5);this.box(1.5,2.3,1.2,white,8.2,1.2,-4.4);this.box(1.55,.1,1.25,edge,8.2,2.4,-4.4);
  for(let i=0;i<7;i++)this.box(1.22,.045,.04,dark,8.2,.5+i*.18,-3.775);
  this.box(.45,.3,.04,this.material(0x58b6a6,.2,.25),8.2,2,-3.77);this.box(.75,1.35,1.15,dark,9.7,.72,-4.4);
  for(let j=0;j<3;j++)this.mesh(new THREE.CylinderGeometry(.08,.08,.35,12),steel,9.48+j*.18,1.58,-4.4);
  // Four charging bays and proportioned passenger vehicles.
  this.chargeLights=[];
  for(let i=0;i<4;i++){
    const x=-7.4+i*4.15;
    for(const side of [-1,1])this.box(.04,.025,4.7,white,x+side*1.45,-.08,5.5);
    this.box(2.9,.025,.04,white,x,-.08,7.85);
    this.car(x,0,5.7,i);
    this.box(.42,1.85,.36,white,x+1.13,.83,3.4);this.box(.3,.65,.06,dark,x+1.13,1.1,3.605);
    const light=this.box(.24,.06,.07,new THREE.MeshStandardMaterial({color:0x62f5cf,emissive:0x39e1b4,emissiveIntensity:1}),x+1.13,1.65,3.605);this.chargeLights.push(light);
    this.tube([[x+1.14,1.2,3.66],[x+1.48,.35,3.94],[x+.8,.37,4.65],[x+.7,.75,4.9]],0x293c47,.04);
  }
  // Utility connection at the campus boundary.
  this.box(1.2,.2,1.3,concrete,9.8,0,5.2);this.box(.2,5.3,.2,steel,9.8,2.65,5.2);this.box(2.4,.15,.15,steel,9.8,4.9,5.2);
  for(let i=0;i<3;i++){this.mesh(new THREE.CylinderGeometry(.08,.08,.25,10),white,9+i*.8,5.1,5.2);this.tube([[9+i*.8,5.25,5.2],[10+i*.8,4.7,7],[12+i*.8,5.4,9]],0x50697b,.012);}
  // Restrained plantings and bollards ground the scale model in an architectural setting.
  for(const x of [-10.8,10.7])for(const z of [-7.5,-.1,7.7]){this.box(.52,.38,.52,concrete,x,.06,z);const shrub=this.mesh(new THREE.IcosahedronGeometry(.38,1),this.material(0x3e6057),x,.5,z);shrub.scale.y=.7;}
  for(let x=-10;x<10;x+=4)this.mesh(new THREE.CylinderGeometry(.045,.045,.4,8),steel,x,.1,8.2);
  const paths=[{p:[[-5,1.12,-.7],[-5,.18,.35],[4.6,.18,.35],[8.2,.5,-3.6]],c:0xefbd80,type:'solar'},
    {p:[[8.2,.5,-3.6],[7.8,.2,.35],[.6,.2,2.7],[-5.9,.25,3.4]],c:0x5ddfc7,type:'ev'},
    {p:[[9.8,.4,5.2],[9.8,.18,1],[8.2,.4,-3.6]],c:0x969fea,type:'grid'}];
  for(const path of paths){const curve=this.tube(path.p,path.c,.025);const particles=[];for(let i=0;i<9;i++){particles.push(this.mesh(new THREE.SphereGeometry(.075,8,6),new THREE.MeshBasicMaterial({color:path.c})));}this.flows.push({curve,particles,type:path.type});}
  // Sun-position arc is an explanatory overlay, not a verified-site sun tracker.
  const arc=[];for(let j=0;j<=60;j++){const a=j/60*Math.PI;arc.push(new THREE.Vector3(Math.cos(a)*12,Math.sin(a)*8+1,-9));}
  const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints(arc),new THREE.LineDashedMaterial({color:0xa28057,dashSize:.22,gapSize:.25,transparent:true,opacity:.4}));line.computeLineDistances();this.scene.add(line);
  this.sunMarker=this.mesh(new THREE.SphereGeometry(.23,20,16),new THREE.MeshBasicMaterial({color:0xfbd99b}),-4,9,-9);
  this.label('PV ARRAYS',[-3.5,2.5,-5]);this.label('POWER CONVERSION',[8.3,3.1,-4.3]);this.label('EV CHARGING',[-1.4,1.7,6.9]);this.label('GRID CONNECTION',[9.7,5.7,5.3]);
 }
 car(x,y,z,index){
  const g=new THREE.Group();g.position.set(x,y,z);this.scene.add(g);
  const paint=this.material([0x9eb9c1,0x567a88,0xb9a992,0x789486][index],.65,.28),glass=this.material(0x172d3c,.7,.15),tire=this.material(0x0b141c,.02,.8),rim=this.material(0x9faeb4,.85,.25);
  // Beveled hull yields mature automotive proportions rather than toy-like cubes.
  const shape=new THREE.Shape();shape.moveTo(-.78,-1.5);shape.lineTo(.78,-1.5);shape.quadraticCurveTo(.92,-1.5,.92,-1.25);shape.lineTo(.92,1.4);shape.quadraticCurveTo(.85,1.65,.6,1.65);shape.lineTo(-.6,1.65);shape.quadraticCurveTo(-.92,1.6,-.92,1.35);shape.lineTo(-.92,-1.25);shape.quadraticCurveTo(-.92,-1.5,-.78,-1.5);
  const body=this.mesh(new THREE.ExtrudeGeometry(shape,{depth:.42,bevelEnabled:true,bevelSegments:3,steps:1,bevelSize:.08,bevelThickness:.08}),paint,0,.43,0,g);body.rotation.x=-Math.PI/2;
  const cabin=this.box(1.48,.48,1.65,glass,0,1.02,0,g);this.box(1.37,.055,1.04,paint,0,1.29,.13,g);
  this.box(1.5,.07,.22,paint,0,.94,-.86,g);this.box(1.45,.06,.3,paint,0,.95,.85,g);
  for(const wx of [-.92,.92])for(const wz of [-.98,1.01]){
   const wheel=this.mesh(new THREE.CylinderGeometry(.29,.29,.19,20),tire,wx,.29,wz,g);wheel.rotation.z=Math.PI/2;
   const hub=this.mesh(new THREE.CylinderGeometry(.17,.17,.2,12),rim,wx,.29,wz,g);hub.rotation.z=Math.PI/2;
  }
  for(const sx of [-.6,.6]){this.box(.34,.085,.055,new THREE.MeshStandardMaterial({color:0xd8f5ff,emissive:0x8bcadb,emissiveIntensity:.7}),sx,.66,-1.57,g);this.box(.3,.055,.05,new THREE.MeshStandardMaterial({color:0x8c4035,emissive:0x892a1e,emissiveIntensity:.3}),sx,.66,1.7,g);}
  for(const sx of [-1,1])this.box(.15,.09,.24,paint,sx,.94,-.47,g);
 }
 setData({solar=.6,ev=.35,grid=.15,zenith=45,azimuth=160}={}){this.ratio=Math.max(0,Math.min(1,solar));this.evRatio=Math.max(0,Math.min(1,ev));this.gridRatio=Math.max(0,Math.min(1,grid));this.sunValues={zenith,azimuth};}
 focus(mode){if(this.failed)return;const views={campus:[[18,20,24],[0,.1,0]],solar:[[9,17,14],[-2,0,-4]],ev:[[12,10,20],[-1,0,4]]};const [p,t]=views[mode]||views.campus;this.destination={p:new THREE.Vector3(...p),t:new THREE.Vector3(...t)};}
 resize(){if(this.failed)return;const w=this.host.clientWidth,h=this.host.clientHeight;this.renderer.setSize(w,h);this.camera.aspect=w/h;this.camera.updateProjectionMatrix();}
 draw(t){if(this.failed)return;const dt=Math.min(.05,(t-(this.last||t))/1000);this.last=t;if(!this.paused)this.clock+=dt;
  if(this.destination){this.camera.position.lerp(this.destination.p,.06);this.controls.target.lerp(this.destination.t,.06);if(this.camera.position.distanceTo(this.destination.p)<.02)this.destination=null;}this.controls.update();
  for(const flow of this.flows){const level=flow.type==='solar'?this.ratio:flow.type==='ev'?this.evRatio:this.gridRatio;flow.particles.forEach((p,i)=>{p.visible=level>.01;p.position.copy(flow.curve.getPoint((this.clock*(.035+level*.09)+i/9)%1));p.scale.setScalar(.5+level*.7);});}
  this.chargeLights.forEach((l,i)=>l.material.emissiveIntensity=this.evRatio>0?(.4+this.evRatio*(.8+.4*Math.sin(this.clock*2+i))):.1);
  const a=(this.sunValues.azimuth-90)/180*Math.PI;const elev=(90-this.sunValues.zenith)*Math.PI/180;this.sunMarker.position.set(Math.cos(a)*10,Math.max(.6,Math.sin(elev)*8+1),-9);this.sunlight.intensity=.8+this.ratio*2.4;
  const w=this.host.clientWidth,h=this.host.clientHeight;for(const tag of this.tags){const p=tag.p.clone().project(this.camera);tag.el.style.left=((p.x*.5+.5)*w)+'px';tag.el.style.top=((-p.y*.5+.5)*h)+'px';tag.el.style.display=Math.abs(p.x)>1||Math.abs(p.y)>1?'none':'';}
  this.renderer.render(this.scene,this.camera);
 }
}
