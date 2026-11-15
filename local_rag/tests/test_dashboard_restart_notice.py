"""Exercise shipped dashboard polling/rendering with a minimal SDK host."""
from pathlib import Path
import shutil
import subprocess
import pytest


@pytest.mark.parametrize('folder', ['src', 'dist'])
def test_dependency_restart_notice_survives_clearing_status_message(folder):
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is required for dashboard behavior checks')
    script = r"""
const fs=require('fs'), vm=require('vm'), assert=require('assert');
const states=[], effects=[]; let cursor=0, Page, poll;
const status={config:{embedding_dimensions:512},hf:{authenticated:false},dependencies_installed:true};
const completed={id:'deps',kind:'dependencies',state:'complete',restart_required:true,detail:'Restart Hermes before continuing. Recovery instructions.'};
const hooks={
 useState(initial){const i=cursor++;if(!(i in states))states[i]=i===2?{id:'deps',state:'running'}:initial;return [states[i],v=>states[i]=typeof v==='function'?v(states[i]):v];},
 useEffect(fn){effects.push(fn);}, useCallback(fn){return fn;},useRef(v){return {current:v};}
};
const window={__HERMES_PLUGINS__:{register(name,fn){Page=fn;}},
 __HERMES_PLUGIN_SDK__:{hooks,React:{createElement:(type,props,...children)=>({type,props,children})},components:new Proxy({},{get:(o,k)=>k}),fetchJSON:url=>Promise.resolve(url.includes('/progress?')?completed:status)},
 setInterval(fn){poll=fn;return 1;},clearInterval(){}};
vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8'),{window});
(async()=>{
 Page(); effects.forEach(fn=>fn()); poll(); await new Promise(setImmediate);
 assert(states[3].text.includes(completed.detail),'completion details were discarded');
 states[3]=null;cursor=0;const tree=Page();
 function alerts(n){if(!n||typeof n!=='object')return [];return [...(n.props?.role==='alert'?[n]:[]),...(n.children||[]).flatMap(alerts)];}
 assert(alerts(tree).some(n=>JSON.stringify(n).includes('Restart Hermes')),'restart-required warning disappeared');
})().catch(e=>{console.error(e);process.exitCode=1;});
"""
    path = Path(__file__).parents[1] / 'dashboard' / folder / 'index.js'
    subprocess.run([node, '-e', script, str(path)], check=True, timeout=15)
