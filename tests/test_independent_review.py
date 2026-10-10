"""Independent acceptance checks: local transports only; no live API requests."""
import asyncio
import base64
import copy
from io import BytesIO
import json
from pathlib import Path
from unittest.mock import patch, AsyncMock

import httpx
from openai import AsyncOpenAI
from PIL import Image
import pytest

from utils.config import ExpConfig
from utils.paperviz_processor import PaperVizProcessor
from utils import generation_utils as gu
from utils.runtime import CandidateCheckpoints, generation_binding, prepare_candidate
from agents.planner_agent import PlannerAgent
from agents.stylist_agent import StylistAgent
from agents.visualizer_agent import VisualizerAgent
from agents.critic_agent import CriticAgent
from agents.retriever_agent import RetrieverAgent
from agents.vanilla_agent import VanillaAgent
from agents.polish_agent import PolishAgent
from rfs import export_bundle

import utils
ROOT = Path(utils.__file__).resolve().parents[1]
SPEC = json.loads((ROOT / 'rfs/examples/encoder-pipeline.figure-spec.json').read_text())


def png(color=(50,80,100,64)):
    f = BytesIO()
    Image.new('RGBA', (18,12), color).save(f, 'PNG')
    return f.getvalue()


def conf(tmp_path, **kw):
    fixture_config=tmp_path / 'offline-configuration.yaml'
    fixture_config.write_text('defaults: {}\n')
    params = dict(dataset_name='independent-fixture', task_name='diagram', exp_mode='demo_full',
                  render_mode='rfs', retrieval_setting='none', main_model_name='gpt-4.1',
                  main_model_provider='openai', image_gen_model_name='gpt-image-1.5',
                  image_gen_provider='openai', max_critic_rounds=1, work_dir=ROOT,
                  checkpoint_dir=tmp_path / 'cp', config_path=fixture_config)
    params.update(kw)
    return ExpConfig(**params)


def processor(config):
    return PaperVizProcessor(config, **{f'{name}_agent': klass(exp_config=config) for name,klass in (
        ('vanilla',VanillaAgent),('planner',PlannerAgent),('stylist',StylistAgent),
        ('visualizer',VisualizerAgent),('critic',CriticAgent),('retriever',RetrieverAgent),('polish',PolishAgent))})


def input_record(**kw):
    return dict(content=SPEC['sources'][0]['content'], visual_intent='Show encoder pipeline.',
                caption='Encoder pipeline', figure_spec=copy.deepcopy(SPEC),
                additional_info={'rounded_ratio':'1:1','image_size':'1K'}, **kw)


def handler_for(requests, *, fail_edit=False):
    def handle(request):
        requests.append(request)
        if request.url.path.endswith('/chat/completions'):
            payload=json.loads(request.content)
            system=payload['messages'][0]['content']
            if 'sanity check' in system:
                ids=[x['id'] for c in ('nodes','edges','groups','labels','datasets') for x in SPEC[c]]
                target=ids[0]
                text=json.dumps({'critic_suggestions':'Improve first-node padding.',
                    'revised_description':'Detailed composition with larger first-node padding and unchanged encoder structure.',
                    'repair':{'instruction':'Increase padding only.','target_ids':[target],
                              'allowed_changes':['padding'],'protected_ids':[x for x in ids if x != target],
                              'protected_regions':[]}})
            else:
                text=json.dumps({'figure_spec':SPEC,'design_description':'Detailed left-to-right encoder design with exact source labels, gray frame, aligned connectors and generous spacing.'})
            return httpx.Response(200, json={'id':'controlled-text','model':payload['model'],
                'choices':[{'index':0,'message':{'role':'assistant','content':text},'finish_reason':'stop'}],
                'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}})
        if request.url.path.endswith('/images/edits') and fail_edit:
            return httpx.Response(400,json={'error':{'message':'Controlled invalid edit','type':'invalid_request_error','code':'invalid'}})
        if request.url.path.endswith(('/images/generations','/images/edits')):
            return httpx.Response(200,json={'created':1,'data':[{'b64_json':base64.b64encode(png()).decode()}]})
        raise AssertionError('Unexpected request: '+str(request.url))
    return handle


async def run_controlled(config, fail_edit=False):
    requests=[]
    transport=httpx.MockTransport(handler_for(requests, fail_edit=fail_edit))
    async with httpx.AsyncClient(transport=transport) as http_client:
        client=AsyncOpenAI(api_key='offline-fixture-value',base_url='https://controlled.invalid/v1',http_client=http_client,max_retries=0)
        p=processor(config)
        try:
            with patch.object(gu,'_client',return_value=client):
                results=[r async for r in p.process_queries_batch([input_record()],do_eval=False)]
        finally:
            await p.aclose()
    return results[0], requests


def test_real_five_agent_openai_transport_bundle(tmp_path):
    result,requests=asyncio.run(run_controlled(conf(tmp_path)))
    assert result['status']=='succeeded', result.get('error')
    image_key=result['eval_image_field'].removesuffix('_base64_jpg')
    record=result['rfs_artifacts'][image_key]
    assert image_key=='target_diagram_critic_desc0'
    assert record['figure_spec']==SPEC
    edit=[r for r in requests if r.url.path.endswith('/images/edits')]
    assert len(edit)==1 and png() in edit[0].content
    assert record['prompt'].encode() in edit[0].content
    manifest=export_bundle(result,tmp_path/'bundle')
    assert manifest['status']=='succeeded'
    assert (tmp_path/'bundle'/'prompt.txt').read_text()==record['prompt']
    assert (tmp_path/'bundle'/'image.png').read_bytes()==png()
    assert [r.url.path for r in requests]==['/v1/chat/completions','/v1/chat/completions','/v1/images/generations','/v1/images/generations','/v1/chat/completions','/v1/images/edits']


def test_failed_refinement_preserves_selected_artifact_in_failure_bundle(tmp_path):
    result,requests=asyncio.run(run_controlled(conf(tmp_path),fail_edit=True))
    assert result['status']=='failed'
    assert result.get('eval_image_field')=='target_diagram_stylist_desc0_base64_jpg'
    manifest=export_bundle(result,tmp_path/'failed-bundle')
    assert manifest['status']=='failed' and 'image' in manifest['files']
    assert (tmp_path/'failed-bundle'/'image.png').read_bytes()==png()


def test_visualizer_discovers_rounds_beyond_config_default(tmp_path):
    config=conf(tmp_path,max_critic_rounds=3,render_mode='legacy')
    v=VisualizerAgent(exp_config=config)
    data={'target_diagram_critic_desc3':'New design','target_diagram_critic_suggestions3':'Improve spacing',
          'additional_info':{'rounded_ratio':'1:1','image_size':'1K'}}
    with patch.object(gu,'call_image_with_retry_async',AsyncMock(return_value=[base64.b64encode(png()).decode()])):
        output=asyncio.run(v.process(data))
    assert 'target_diagram_critic_desc3_base64_jpg' in output


def test_corrupt_success_checkpoint_is_not_returned_as_success(tmp_path):
    config=conf(tmp_path,max_critic_rounds=0)
    first,_=asyncio.run(run_controlled(config))
    path=CandidateCheckpoints(config.checkpoint_dir).path_for(first['candidate_id'])
    saved=json.loads(path.read_text())
    selected=saved['result']['eval_image_field']
    saved['result'][selected]='Error'
    saved['result']['rfs_artifacts']={}
    path.write_text(json.dumps(saved))
    resumed,requests=asyncio.run(run_controlled(conf(tmp_path,max_critic_rounds=0,resume=True)))
    assert resumed['status']=='succeeded'
    assert resumed.get('resumed') is not True, 'Corrupt success checkpoint must be a cache miss and rerun'
    assert requests, 'Invalid checkpoint must not skip generation'


def test_benchmark_candidate_id_cannot_escape_bundle_root(tmp_path):
    import main as benchmark
    raw=input_record(candidate_id='../escaped-candidate')
    source=tmp_path/'input.json'
    source.write_text(json.dumps([raw]))
    cfg=tmp_path/'config.yaml'
    cfg.write_text('defaults: {}\n')
    bundle_root=tmp_path/'bundles'
    argv=['--input-file',str(source),'--output-file',str(tmp_path/'results.json'),
          '--config',str(cfg),'--retrieval-setting','none','--no-eval',
          '--main-model-name','gpt-4.1','--main-model-provider','openai',
          '--image-gen-model-name','gpt-image-1.5','--image-gen-provider','openai',
          '--render-mode','rfs','--rfs-output-dir',str(bundle_root),'--max-critic-rounds','0']
    async def run():
        requests=[]
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler_for(requests))) as http_client:
            client=AsyncOpenAI(api_key='offline-fixture-value',base_url='https://controlled.invalid/v1',http_client=http_client,max_retries=0)
            with patch.object(gu,'_client',return_value=client), patch.object(gu,'validate_generation_configuration',return_value={}):
                try:
                    await benchmark.main(argv)
                except (ValueError,RuntimeError):
                    pass  # Valid to reject an unsafe input before generation.
    asyncio.run(run())
    assert not (tmp_path/'escaped-candidate').exists(), 'Input candidate_id traversed configured bundle output root'


@pytest.mark.parametrize('module_name',['app','demo'])
def test_ui_created_candidates_reach_real_runtime(tmp_path,module_name,monkeypatch):
    import importlib
    monkeypatch.setenv('GRADIO_ANALYTICS_ENABLED','False')
    original_exists=Path.exists
    monkeypatch.setattr(Path, 'exists', lambda path: False if path == ROOT / 'configs' / 'model_config.yaml' else original_exists(path))
    module=importlib.import_module(module_name)
    records=module.create_sample_inputs('Real controlled source.','Show source pipeline.',num_copies=1,max_critic_rounds=0,aspect_ratio='1:1',figure_size='1K')
    config=conf(tmp_path,render_mode='legacy',max_critic_rounds=0,main_model_name='gpt-4.1',image_gen_model_name='gpt-image-1.5')
    p=processor(config)
    async def run():
        requests=[]
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler_for(requests))) as http_client:
            client=AsyncOpenAI(api_key='offline-fixture-value',base_url='https://controlled.invalid/v1',http_client=http_client,max_retries=0)
            with patch.object(gu,'_client',return_value=client):
                return [r async for r in p.process_queries_batch(records,do_eval=False)]
    try:
        results=asyncio.run(run())
    finally:
        asyncio.run(p.aclose())
    assert len(results)==1 and results[0]['status']=='succeeded'


def test_timed_out_candidates_do_not_accumulate_unbounded_retrieval_requests(tmp_path):
    class SlowRetriever:
        active=0
        maximum=0
        async def process(self,data,**kw):
            self.active+=1
            self.maximum=max(self.maximum,self.active)
            try:
                await asyncio.sleep(5)
                return data
            finally:
                self.active-=1
    tracker=SlowRetriever()
    config=conf(tmp_path,max_concurrent=2,candidate_timeout_seconds=0.02)
    p=processor(config)
    p.retriever_agent=tracker
    records=[]
    for i in range(12):
        item=input_record()
        item['content']=f'Distinct source {i}'
        records.append(item)
    async def run():
        try:
            results=[r async for r in p.process_queries_batch(records,do_eval=False)]
            assert len(results)==12 and all(r['status']=='failed' for r in results)
            assert tracker.active==0
            assert tracker.maximum<=2, f'{tracker.maximum} simultaneous retrieval calls despite max_concurrent=2'
        finally:
            await p.aclose()
    asyncio.run(run())


@pytest.mark.parametrize('finish_reason',['stop','length','content_filter'])
def test_openrouter_sdk_payload_and_terminal_status(tmp_path,finish_reason):
    requests=[]
    raw=png()
    model='openrouter/google/gemini-nano-banana-2.1'
    def handler(request):
        requests.append(request)
        body=json.loads(request.content)
        assert body['model']=='google/gemini-nano-banana-2.1'
        assert body['modalities']==['image','text']
        assert body['image_config']=={'aspect_ratio':'1:1','image_size':'1K'}
        image_part=next(x for x in body['messages'][1]['content'] if x['type']=='image_url')
        assert image_part['image_url']['url']=='data:image/png;base64,'+base64.b64encode(raw).decode()
        return httpx.Response(200,json={'id':'controlled-gateway','model':body['model'],
            'choices':[{'index':0,'message':{'role':'assistant','content':'',
                         'images':[{'image_url':{'url':'data:image/png;base64,'+base64.b64encode(raw).decode()}}]},
                        'finish_reason':finish_reason}]})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client=AsyncOpenAI(api_key='offline-fixture-value',base_url='https://controlled.invalid/v1',http_client=http_client,max_retries=0)
            with patch.object(gu,'_client',return_value=client):
                call=gu.generate_images(model,[{'type':'text','text':'Bounded change only.'},
                    {'type':'image','data':base64.b64encode(raw).decode(),'mime_type':'image/png'}],
                    {'aspect_ratio':'1:1','image_size':'1K'},operation='image_edit',max_attempts=1)
                if finish_reason=='stop':
                    output=await call
                    assert output.samples[0].image_mime=='image/png'
                else:
                    with pytest.raises(gu.GenerationResponseError):
                        await call
    asyncio.run(run())
    assert len(requests)==1


def test_real_gemini_sdk_transport_five_agent_bundle(tmp_path):
    from google import genai
    from google.genai import types
    requests=[]
    def handler(request):
        requests.append(request)
        body=json.loads(request.content)
        system=' '.join(x.get('text','') for x in body.get('systemInstruction',{}).get('parts',[]))
        if 'sanity check' in system:
            text=json.dumps({'critic_suggestions':'No changes needed.','revised_description':'No changes needed.','repair':None})
            parts=[{'text':text}]
        elif 'responseModalities' in body.get('generationConfig',{}):
            parts=[{'inlineData':{'mimeType':'image/png','data':base64.b64encode(png()).decode()}}]
            assert body['generationConfig']['imageConfig']=={'aspectRatio':'1:1','imageSize':'1K'}
            assert body['generationConfig']['candidateCount']==1
        else:
            text=json.dumps({'figure_spec':SPEC,'design_description':'Controlled detailed encoder layout, consistent typography, source-exact terminology and spaced connectors.'})
            parts=[{'text':text}]
        return httpx.Response(200,json={'candidates':[{'content':{'role':'model','parts':parts},'finishReason':'STOP'}],
                                       'responseId':'controlled-gemini'})
    config=conf(tmp_path,main_model_name='gemini-3.8-flash',main_model_provider='gemini',
                image_gen_model_name='gemini-nano-banana-2.1',image_gen_provider='gemini')
    async def run():
        client=genai.Client(vertexai=False,api_key='offline-fixture-value',
            http_options=types.HttpOptions(base_url='https://controlled.invalid',
                async_client_args={'transport':httpx.MockTransport(handler)},
                retry_options=types.HttpRetryOptions(attempts=1)))
        p=processor(config)
        try:
            with patch.object(gu,'_client',return_value=client):
                results=[r async for r in p.process_queries_batch([input_record()],do_eval=False)]
            assert results[0]['status']=='succeeded',results[0].get('error')
            manifest=export_bundle(results[0],tmp_path/'gemini-bundle')
            selected=results[0]['eval_image_field'].removesuffix('_base64_jpg')
            actual=results[0]['rfs_artifacts'][selected]
            assert (tmp_path/'gemini-bundle'/'image.png').read_bytes()==png()
            assert (tmp_path/'gemini-bundle'/'prompt.txt').read_text()==actual['prompt']
            assert manifest['provider']=='gemini'
            image_requests=[json.loads(r.content) for r in requests if 'imageConfig' in json.loads(r.content).get('generationConfig',{})]
            assert len(image_requests)==2
            assert image_requests[-1]['contents'][0]['parts'][0]['text']==actual['prompt']
            assert image_requests[-1]['systemInstruction']['parts'][0]['text']==actual['system_prompt']
        finally:
            await p.aclose()
            await client.aio.aclose()
            client.close()
    asyncio.run(run())
    assert len(requests)==5


def test_httpx2_transient_error_is_bounded_and_safe():
    import httpx2
    from utils.model_registry import resolve_model
    profile=resolve_model('gemini-3.8-flash')
    calls=[]
    async def request():
        calls.append(None)
        if len(calls)==1:
            raise httpx2.ConnectError('Do not expose this controlled fake credential.')
        return 'controlled-result'
    value,attempts=asyncio.run(gu._retry(request,profile,max_attempts=3,retry_delay=0,timeout=0.1,total_timeout=0.5))
    assert value=='controlled-result' and attempts==2 and len(calls)==2
    error=gu._classify(httpx2.ConnectError('Do not expose this controlled fake credential.'),profile)
    assert error.retryable and 'credential' not in str(error)


@pytest.mark.parametrize('provider',['gemini','openrouter'])
def test_no_system_capability_actual_wire_matches_recorded_prompt(tmp_path,provider):
    from google import genai
    from google.genai import types
    requests=[]
    model='gemini-nano-banana-2.1' if provider=='gemini' else 'openrouter/google/gemini-nano-banana-2.1'
    contents=[{'type':'text','text':'Controlled prompt one.'},{'type':'text','text':'Controlled prompt two.'},
              {'type':'image','data':base64.b64encode(png()).decode(),'mime_type':'image/png'}]
    original=copy.deepcopy(contents)
    expected='Controlled separate system.\n\nControlled prompt one.\nControlled prompt two.'
    def handler(request):
        requests.append(request)
        body=json.loads(request.content)
        if provider=='gemini':
            actual='\n'.join(x['text'] for x in body['contents'][0]['parts'] if 'text' in x)
            assert not body.get('systemInstruction')
            assert actual==expected
            assert any('inlineData' in x for x in body['contents'][0]['parts'])
            return httpx.Response(200,json={'candidates':[{'content':{'role':'model','parts':[
                {'inlineData':{'mimeType':'image/png','data':base64.b64encode(png()).decode()}}]},'finishReason':'STOP'}]})
        assert all(x['role']!='system' for x in body['messages'])
        actual='\n'.join(x['text'] for x in body['messages'][0]['content'] if x['type']=='text')
        assert actual==expected
        assert any(x['type']=='image_url' for x in body['messages'][0]['content'])
        return httpx.Response(200,json={'id':'controlled-no-system','model':body['model'],'choices':[{
            'index':0,'message':{'role':'assistant','content':'','images':[{'image_url':{
                'url':'data:image/png;base64,'+base64.b64encode(png()).decode()}}]},'finish_reason':'stop'}]})
    async def run():
        if provider=='gemini':
            client=genai.Client(vertexai=False,api_key='offline-fixture-value',http_options=types.HttpOptions(
                base_url='https://controlled.invalid',async_client_args={'transport':httpx.MockTransport(handler)},
                retry_options=types.HttpRetryOptions(attempts=1)))
            try:
                with patch.object(gu,'_client',return_value=client):
                    result=await gu.generate_images(model,contents,{'system_prompt':'Controlled separate system.','image_size':'1K'},
                        overrides={model:{'supports_system_prompt':False}},operation='image_edit',max_attempts=1)
            finally:
                await client.aio.aclose()
                client.close()
        else:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
                client=AsyncOpenAI(api_key='offline-fixture-value',base_url='https://controlled.invalid/v1',http_client=http_client,max_retries=0)
                with patch.object(gu,'_client',return_value=client):
                    result=await gu.generate_images(model,contents,{'system_prompt':'Controlled separate system.','image_size':'1K'},
                        overrides={model:{'supports_system_prompt':False}},operation='image_edit',max_attempts=1)
        assert result.effective_prompt==expected and result.effective_system_prompt is None
        assert contents==original
    asyncio.run(run())
    assert len(requests)==1
