import http from 'node:http';
import {readFile, mkdir, writeFile, readdir} from 'node:fs/promises';
import path from 'node:path';
import {randomUUID} from 'node:crypto';
const root=process.cwd(), data=path.join(root,'data');
const mime={'.html':'text/html; charset=utf-8','.js':'text/javascript','.mjs':'text/javascript','.css':'text/css','.jpg':'image/jpeg','.png':'image/png','.wasm':'application/wasm','.json':'application/json'};
await mkdir(data,{recursive:true});
http.createServer(async(req,res)=>{
 try{
  const url=new URL(req.url,'http://localhost');
  if(url.pathname==='/api/samples' && req.method==='GET'){
   const files=(await readdir(data)).filter(x=>x.endsWith('.json'));
   const counts={}; for(const file of files){const s=JSON.parse(await readFile(path.join(data,file),'utf8'));counts[s.label]=(counts[s.label]||0)+1;}
   res.setHeader('Content-Type','application/json');res.end(JSON.stringify({total:files.length,counts}));return;
  }
  if(url.pathname==='/api/samples' && req.method==='POST'){
   if(req.headers.origin && !['http://localhost:5173','http://127.0.0.1:5173'].includes(req.headers.origin)){res.writeHead(403);res.end();return;}
   let chunks=[],size=0;for await(const c of req){size+=c.length;if(size>12000000){res.writeHead(413);res.end();return;}chunks.push(c);}
   const s=JSON.parse(Buffer.concat(chunks));
   if(!Number.isInteger(s.label)||s.label<0||s.label>13||!Array.isArray(s.frames)||s.frames.length<5||s.frames.length>200||typeof s.person!=='string'||s.person.length>80){res.writeHead(400);res.end('Invalid sample');return;}
   const id=randomUUID();await writeFile(path.join(data,id+'.json'),JSON.stringify({...s,id,savedAt:new Date().toISOString(),schemaVersion:1}));
   res.setHeader('Content-Type','application/json');res.end(JSON.stringify({id}));return;
  }
  if(req.method!=='GET'){res.writeHead(405);res.end();return;}
  const p=decodeURIComponent(url.pathname);
  const base=p.startsWith('/vendor/')?path.join(root,'node_modules','@mediapipe','tasks-vision'):path.join(root,'public');
  const relative=p.startsWith('/vendor/')?p.slice(8):p==='/'?'index.html':p.slice(1);
  const file=path.resolve(base,relative);
  if(!file.startsWith(base+path.sep)){res.writeHead(403);res.end();return;}
  res.setHeader('Content-Type',mime[path.extname(file)]||'application/octet-stream');res.end(await readFile(file));
 }catch(e){res.writeHead(e instanceof SyntaxError?400:404);res.end('Request failed');}
}).listen(5173,'127.0.0.1',()=>console.log('Meme Matcher · 9.5AI Club: http://localhost:5173'));
