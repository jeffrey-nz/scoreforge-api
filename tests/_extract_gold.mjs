import http from 'http';
import fs from 'fs';
const JOB='f58440f9-b218-4f99-b606-ec95094223b0';
function get(path){return new Promise((res,rej)=>{http.get({host:'localhost',port:8001,path},x=>{let s='';x.on('data',c=>s+=c);x.on('end',()=>res(s));}).on('error',rej);});}
const j = JSON.parse(await get(`/pipeline/jobs/${JOB}`));
const bars = j.bars || [];
const meta = j.meta || {};
const verified = bars.filter(b => b.verified);
console.log('total bars:', bars.length, ' verified:', verified.length);
const byPage = {};
for (const b of verified) (byPage[b.page] ||= []).push(b.n);
for (const p of Object.keys(byPage).sort((a,b)=>a-b)) console.log(`  page ${p}: bars ${byPage[p].join(',')}`);
// Gold standard = the user's hand-VERIFIED bars; keep the musical fields.
const gold = verified.map(b => ({
  n: b.n, page: b.page, timeSig: b.timeSig || meta.timeSig || '3/8',
  melody: b.melody||'', melody2: b.melody2||'', bass: b.bass||'', bass2: b.bass2||'',
  clef_changes: b.clef_changes||null,
}));
fs.writeFileSync('tests/fixtures/fur_elise_gold.json', JSON.stringify({ meta:{key:meta.key,timeSig:meta.timeSig,bpm:meta.bpm,title:meta.title}, bars: gold }, null, 2));
console.log('wrote', gold.length, 'gold bars -> tests/fixtures/fur_elise_gold.json');
