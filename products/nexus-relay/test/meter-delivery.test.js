import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { Miniflare } from "miniflare";
import { enqueueUsageForBilling, flushBillingOutbox } from "../src/meter_delivery.js";

async function database() {
  const mf = new Miniflare({ modules:true, script:"export default {fetch(){return new Response('ok')}}", d1Databases:{DB:"00000000-0000-0000-0000-000000000003"} });
  const db = await mf.getD1Database("DB");
  const schema = await readFile(new URL("../schema.sql", import.meta.url), "utf8");
  for (const statement of schema.split(";").map(x=>x.trim()).filter(Boolean)) await db.prepare(statement).run();
  return {mf,db};
}

test("successful usage emits idempotent Stripe meter event", async()=>{
  const {mf,db}=await database();
  try{
    const calls=[];
    const env={DB:db,STRIPE_SECRET_KEY:"sk_test",STRIPE_FETCH:async(url,options)=>{calls.push({url,body:String(options.body)});return new Response(JSON.stringify({identifier:"usage_1"}),{status:200,headers:{"content-type":"application/json"}});},RELAY_PRICE_CATALOG:JSON.stringify({price_team:{plan:"team",meter_event_name:"relay_reads"}})};
    const principal={customer_id:"cus_1",stripe_customer_id:"cus_stripe_1",plan:"team"};
    const result=await enqueueUsageForBilling(env,principal,"usage_1");
    assert.equal(result.status,"sent");
    assert.equal(calls.length,1);
    assert.match(calls[0].body,/identifier=usage_1/);
    assert.match(calls[0].body,/event_name=relay_reads/);
    const row=await db.prepare("SELECT status,attempts FROM billing_outbox WHERE id='usage_1'").first();
    assert.equal(row.status,"sent");assert.equal(row.attempts,1);
  }finally{await mf.dispose();}
});

test("failed delivery remains pending and scheduled flush retries", async()=>{
  const {mf,db}=await database();
  try{
    let fail=true;
    const env={DB:db,STRIPE_SECRET_KEY:"sk_test",STRIPE_METER_EVENT_NAME:"relay_reads",STRIPE_FETCH:async()=> fail ? new Response(JSON.stringify({error:{message:"temporary"}}),{status:500,headers:{"content-type":"application/json"}}) : new Response(JSON.stringify({ok:true}),{status:200,headers:{"content-type":"application/json"}})};
    const principal={customer_id:"cus_2",stripe_customer_id:"cus_stripe_2",plan:"developer"};
    const first=await enqueueUsageForBilling(env,principal,"usage_2");
    assert.equal(first.status,"pending");
    fail=false;
    const flushed=await flushBillingOutbox(env);
    assert.equal(flushed.sent,1);
    const row=await db.prepare("SELECT status,attempts FROM billing_outbox WHERE id='usage_2'").first();
    assert.equal(row.status,"sent");assert.equal(row.attempts,2);
  }finally{await mf.dispose();}
});
