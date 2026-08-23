export const SHOWCASE_SCENARIOS = [
  {
    id: 'research',
    title: 'Research synthesis',
    prompt: 'Compare three sources on a technical claim, identify contradictions, and produce a receipt-ready synthesis.',
    capability: 'skill.research',
  },
  {
    id: 'browser',
    title: 'Chrome task planning',
    prompt: 'Plan a browser task that opens a product page, extracts structured facts, and stops before any purchase or account mutation.',
    capability: 'browser.task.enqueue',
  },
  {
    id: 'iot',
    title: 'IoT governance',
    prompt: 'Given a robot command proposal, explain the identity, approval, replay, deadline, and receipt checks required before execution.',
    capability: 'skill.reason',
  },
  {
    id: 'brain',
    title: 'BRAIN-AI state mechanics',
    prompt: 'Design a deterministic state transition with optimistic locking, TTL leases, circuit breakers, and structured telemetry.',
    capability: 'brain.cycle',
  },
];

export function buildShowcasePrompt(basePrompt, scenario) {
  return [
    'You are AURO Chrome LLM operating locally in the browser.',
    'Remote model loading is disabled.',
    'Do not claim that an external action happened unless a governed capability receipt proves it.',
    'Separate observation, plan, approval-required action, execution result, and evidence.',
    `Scenario: ${scenario.title}`,
    `Capability context: ${scenario.capability}`,
    `Task: ${basePrompt || scenario.prompt}`,
  ].join('\n');
}
