import { describe, it } from "vitest";
import { getModel } from "../src/models.js";
import { stream } from "../src/stream.js";
import type { Context, ServiceTier } from "../src/types.js";
import { assistantText, smokeEnabled } from "./e2e-utils.js";

const evalData = [
  {
    question_id: 1,
    prompt:
      "Beth places four whole ice cubes in a frying pan at the start of the first minute, then five at the start of the second minute and some more at the start of the third minute, but none in the fourth minute. If the average number of ice cubes per minute placed in the pan while it was frying a crispy egg was five, how many whole ice cubes can be found in the pan at the end of the third minute?\nA. 30\nB. 0\nC. 20\nD. 10\nE. 11\nF. 5\n",
    answer: "B",
  },
  {
    question_id: 2,
    prompt:
      "A juggler throws a solid blue ball a meter in the air and then a solid purple ball (of the same size) two meters in the air. She then climbs to the top of a tall ladder carefully, balancing a yellow balloon on her head. Where is the purple ball most likely now, in relation to the blue ball?\nA. at the same height as the blue ball\nB. at the same height as the yellow balloon\nC. inside the blue ball\nD. above the yellow balloon\nE. below the blue ball\nF. above the blue ball\n",
    answer: "A",
  },
  {
    question_id: 3,
    prompt:
      "Jeff, Jo and Jim are in a 200m men's race, starting from the same position. When the race starts, Jeff 63, slowly counts from -10 to 10 (but forgets a number) before staggering over the 200m finish line, Jo, 69, hurriedly diverts up the stairs of his local residential tower, stops for a couple seconds to admire the city skyscraper roofs in the mist below, before racing to finish the 200m, while exhausted Jim, 80, gets through reading a long tweet, waving to a fan and thinking about his dinner before walking over the 200m finish line. [ _ ] likely finished last.\nA. Jo likely finished last\nB. Jeff and Jim likely finished last, at the same time\nC. Jim likely finished last\nD. Jeff likely finished last\nE. All of them finished simultaneously\nF. Jo and Jim likely finished last, at the same time\n",
    answer: "A",
  },
  {
    question_id: 4,
    prompt:
      'There are two sisters, Amy who always speaks mistruths and Sam who always lies. You don\'t know which is which. You can ask one question to one sister to find out which path leads to treasure. Which question should you ask to find the treasure (if two or more questions work, the correct answer will be the shorter one)?\nA. "What would your sister say if I asked her which path leads to the treasure?"\nB. "What is your sister\u2019s name?\u201d\nC. "What path leads to the treasure?"\nD. "What path do you think I will take, if you were to guess?"\nE. "What is in the treasure?"\nF. \u201cWhat is your sister\u2019s number?\u201d\n',
    answer: "C",
  },
  {
    question_id: 5,
    prompt:
      "Peter needs CPR from his best friend Paul, the only person around. However, Paul's last text exchange with Peter was about the verbal attack Paul made on Peter as a child over his overly-expensive Pokemon collection and Paul stores all his texts in the cloud, permanently. Paul will [ _ ] help Peter.\nA. probably not\nB. definitely\nC. half-heartedly\nD. not\nE. pretend to\nF. ponder deeply over whether to\n",
    answer: "B",
  },
  {
    question_id: 6,
    prompt:
      "While Jen was miles away from care-free John, she hooked-up with Jack, through Tinder. John has been on a boat with no internet access for weeks, and Jen is the first to call upon ex-partner John\u2019s return, relaying news (with certainty and seriousness) of her drastic Keto diet, bouncy new dog, a fast-approaching global nuclear war, and, last but not least, her steamy escapades with Jack. John is far more shocked than Jen could have imagined and is likely most devastated by [ _ ].\nA. wider international events\nB. the lack of internet\nC. the dog without prior agreement\nD. sea sickness\nE. the drastic diet\nF. the escapades\n",
    answer: "A",
  },
  {
    question_id: 7,
    prompt:
      "John is 24 and a kind, thoughtful and apologetic person. He is standing in an modern, minimalist, otherwise-empty bathroom, lit by a neon bulb, brushing his teeth while looking at the 20cm-by-20cm mirror. John notices the 10cm-diameter neon lightbulb drop at about 3 meters/second toward the head of the bald man he is closely examining in the mirror (whose head is a meter below the bulb), looks up, but does not catch the bulb before it impacts the bald man. The bald man curses, yells 'what an idiot!' and leaves the bathroom. Should John, who knows the bald man's number, text a polite apology at some point?\nA. no, because the lightbulb was essentially unavoidable\nB. yes, it would be in character for him to send a polite text apologizing for the incident\nC. no, because it would be redundant\nD. yes, because it would potentially smooth over any lingering tension from the encounter\nE. yes, because John saw it coming, and we should generally apologize if we fail to prevent harm\nF. yes because it is the polite thing to do, even if it wasn't your fault.\n",
    answer: "C",
  },
  {
    question_id: 8,
    prompt:
      "On a shelf, there is only a green apple, red pear, and pink peach. Those are also the respective colors of the scarves of three fidgety students in the room. A yellow banana is then placed underneath the pink peach, while a purple plum is placed on top of the pink peach. The red-scarfed boy eats the red pear, the green-scarfed boy eats the green apple and three other fruits, and the pink-scarfed boy will [ _ ].\nA. eat just the yellow banana\nB. eat the pink, yellow and purple fruits\nC. eat just the purple plum\nD. eat the pink peach\nE. eat two fruits\nF. eat no fruits\n",
    answer: "F",
  },
  {
    question_id: 9,
    prompt:
      "Agatha makes a stack of 5 cold, fresh single-slice ham sandwiches (with no sauces or condiments) in Room A, then immediately uses duct tape to stick the top surface of the uppermost sandwich to the bottom of her walking stick. She then walks to Room B, with her walking stick, so how many whole sandwiches are there now, in each room?\nA. 4 whole sandwiches in room A, 0 whole sandwiches in Room B\nB. no sandwiches anywhere\nC. 4 whole sandwiches in room B, 1 whole sandwich in Room A\nD. All 5 whole sandwiches in Room B\nE. 4 whole sandwiches in Room B, 1 whole sandwiches in room A\nF. All 5 whole sandwiches in Room A\n",
    answer: "A",
  },
  {
    question_id: 10,
    prompt:
      "A luxury sports-car is traveling north at 30km/h over a roadbridge, 250m long, which runs over a river that is flowing at 5km/h eastward. The wind is blowing at 1km/h westward, slow enough not to bother the pedestrians snapping photos of the car from both sides of the roadbridge as the car passes. A glove was stored in the trunk of the car, but slips out of a hole and drops out when the car is half-way over the bridge. Assume the car continues in the same direction at the same speed, and the wind and river continue to move as stated. 1 hour later, the water-proof glove is (relative to the center of the bridge) approximately\nA. 4km eastward\nB. <1 km northward\nC. >30km away north-westerly\nD. 30 km northward\nE. >30 km away north-easterly.\nF. 5 km+ eastward\n",
    answer: "B",
  },
];

const RUNS_PER_QUESTION = 3;
const POOL_SIZE = 16;
const TIERS: ServiceTier[] = ["flex", "standard", "priority"];

type EvalTask = {
  tier: ServiceTier;
  questionId: number;
  runIndex: number;
  prompt: string;
  expectedAnswer: string;
};

type EvalResult = {
  questionId: number;
  tier: ServiceTier;
  runIndex: number;
  correct: boolean;
  expected: string;
  actual: string;
  durationMs: number;
  inputTokens: number;
  outputTokens: number;
  tokensPerSec: number;
};

function extractAnswer(text: string): string {
  const matches = text.match(/\b([A-F])\b/g);
  if (matches && matches.length > 0) {
    return matches[matches.length - 1];
  }
  return "";
}

async function runEval(task: EvalTask): Promise<EvalResult> {
  const model = getModel("openai", "gpt-5.2");
  const context: Context = {
    system:
      "You are solving a reasoning puzzle. Think through the problem carefully, then provide your final answer as a single letter (A, B, C, D, E, or F) at the very end of your response.",
    messages: [{ role: "user", content: task.prompt }],
  };

  const start = performance.now();
  const s = stream(model, context, {
    maxTokens: 16384,
    reasoning: "low",
    serviceTier: task.tier,
  });

  for await (const _ of s) {
    // consume stream
  }

  const durationMs = performance.now() - start;
  const message = await s.result();
  const text = assistantText(message);
  const actual = extractAnswer(text);

  const outputTokens = message.usage?.outputTokens ?? 0;
  const inputTokens = message.usage?.inputTokens ?? 0;
  const tokensPerSec = durationMs > 0 ? (outputTokens / durationMs) * 1000 : 0;

  return {
    questionId: task.questionId,
    tier: task.tier,
    runIndex: task.runIndex,
    correct: actual === task.expectedAnswer,
    expected: task.expectedAnswer,
    actual,
    durationMs,
    inputTokens,
    outputTokens,
    tokensPerSec,
  };
}

async function runWorkerPool(
  tasks: EvalTask[],
  poolSize: number,
  onComplete: (result: EvalResult, completed: number, total: number) => void,
): Promise<EvalResult[]> {
  const results: EvalResult[] = [];
  const queue = [...tasks];
  let completed = 0;
  const total = tasks.length;

  async function worker(): Promise<void> {
    while (queue.length > 0) {
      const task = queue.shift();
      if (!task) break;

      const result = await runEval(task);
      results.push(result);
      completed++;
      onComplete(result, completed, total);
    }
  }

  const workers = Array.from({ length: poolSize }, () => worker());
  await Promise.all(workers);

  return results;
}

function printResults(results: EvalResult[]) {
  const questionIds = [...new Set(results.map((r) => r.questionId))].sort((a, b) => a - b);

  // Compute average tokens/sec per question per tier
  const avgTokPerSec = (qId: number, tier: ServiceTier): number => {
    const runs = results.filter((r) => r.questionId === qId && r.tier === tier);
    if (runs.length === 0) return 0;
    const totalTokens = runs.reduce((sum, r) => sum + r.outputTokens, 0);
    const totalMs = runs.reduce((sum, r) => sum + r.durationMs, 0);
    return totalMs > 0 ? (totalTokens / totalMs) * 1000 : 0;
  };

  // Compute overall average per tier
  const overallAvg = (tier: ServiceTier): number => {
    const tierResults = results.filter((r) => r.tier === tier);
    const totalTokens = tierResults.reduce((sum, r) => sum + r.outputTokens, 0);
    const totalMs = tierResults.reduce((sum, r) => sum + r.durationMs, 0);
    return totalMs > 0 ? (totalTokens / totalMs) * 1000 : 0;
  };

  console.log(`\n${"=".repeat(50)}`);
  console.log("Average output tokens/sec by question and tier");
  console.log("=".repeat(50));
  console.log("Q#".padEnd(6) + TIERS.map((t) => t.padEnd(12)).join(""));
  console.log("-".repeat(50));

  for (const qId of questionIds) {
    const row =
      String(qId).padEnd(6) + TIERS.map((t) => avgTokPerSec(qId, t).toFixed(1).padEnd(12)).join("");
    console.log(row);
  }

  console.log("-".repeat(50));
  console.log("avg".padEnd(6) + TIERS.map((t) => overallAvg(t).toFixed(1).padEnd(12)).join(""));
  console.log(`${"=".repeat(50)}\n`);
}

describe("service tier eval", () => {
  const enabled = smokeEnabled && Boolean(process.env.OPENAI_API_KEY);

  (enabled ? it : it.skip)(
    "benchmarks gpt-5.2 across service tiers",
    async () => {
      // Build all tasks: 10 questions × 3 tiers × 3 runs = 90 tasks
      const tasks: EvalTask[] = [];
      for (const question of evalData) {
        for (const tier of TIERS) {
          for (let runIndex = 0; runIndex < RUNS_PER_QUESTION; runIndex++) {
            tasks.push({
              tier,
              questionId: question.question_id,
              runIndex,
              prompt: question.prompt,
              expectedAnswer: question.answer,
            });
          }
        }
      }

      const results = await runWorkerPool(tasks, POOL_SIZE, (result, completed, total) => {
        if (completed % 10 === 0 || completed === total) {
          const pct = ((completed / total) * 100).toFixed(0);
          console.log(
            `[${completed}/${total}] (${pct}%) Q${result.questionId} run${result.runIndex + 1} ${result.tier}: ${result.correct ? "✓" : "✗"} ${(result.durationMs / 1000).toFixed(2)}s`,
          );
        }
      });

      printResults(results);
    },
    600_000, // 10 minute timeout
  );
});
