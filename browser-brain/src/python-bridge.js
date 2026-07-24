export class PythonEngineBridge{
  constructor({baseUrl='http://127.0.0.1:8090',apiToken=null,executionToken=null}={}){this.baseUrl=baseUrl.replace(/\/$/,'');this.apiToken=apiToken;this.executionToken=executionToken}
  headers(execution=false){const h={'content-type':'application/json'};if(this.apiToken)h.authorization=`Bearer ${this.apiToken}`;if(execution&&this.executionToken)h['x-auro-execution-token']=this.executionToken;return h}
  async request(path,body,execution=false){const response=await fetch(this.baseUrl+path,{method:body?'POST':'GET',headers:this.headers(execution),body:body?JSON.stringify(body):undefined});const data=await response.json();if(!response.ok)throw new Error(data.error?.message||`Python engine HTTP ${response.status}`);return data}
  async status(){return this.request('/v1/health/ready')}
  async think(message,{execute=false}={}){return this.request('/v1/him/respond',{message,execute},execute)}
  async capability(name,args={},approved=false){return this.request('/v1/capabilities/call',{name,arguments:args,approved},approved)}
  async generateDocuments(spec){return this.capability('office.create_bundle',spec,true)}
  async verifyReceipts(){return this.request('/v1/receipts/verify')}
  async claimBrowserTask(workerId){return this.request('/v1/browser/tasks/claim',{worker_id:workerId})}
  async completeBrowserTask(taskId,{result=null,error=null}={}){return this.request(`/v1/browser/tasks/${taskId}/complete`,{result,error})}
}
