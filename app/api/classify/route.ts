import { NextResponse } from 'next/server';
import { validatePassage } from '@/lib/validation';
import type { ClassificationResult, Score } from '@/lib/types';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

function validScores(value: unknown): value is Score[] {
  return Array.isArray(value) && value.length > 0 && value.length <= 18 && value.every((item) =>
    item && typeof item.label === 'string' && item.label.length > 0 &&
    typeof item.score === 'number' && Number.isFinite(item.score) && item.score >= 0 && item.score <= 1
  );
}

export async function POST(request: Request) {
  let body: unknown;
  try { body = await request.json(); }
  catch { return NextResponse.json({ error: 'Request body must be valid JSON.' }, { status: 400 }); }

  const text = (body && typeof body === 'object' && 'text' in body) ? (body as { text: unknown }).text : undefined;
  const checked = validatePassage(text);
  if (!checked.ok) return NextResponse.json({ error: checked.message }, { status: 400 });

  const upstream = process.env.CLASSIFIER_API_URL?.trim();
  if (!upstream) {
    const result: ClassificationResult = {
      configured: false,
      mode: 'not-configured',
      message: 'The interface is ready, but no trained classifier is connected yet. Add CLASSIFIER_API_URL after deploying your model API to enable real predictions.'
    };
    return NextResponse.json(result, { status: 200, headers: { 'Cache-Control': 'no-store' } });
  }

  try {
    const base = upstream.replace(/\/+$/, '');
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (process.env.CLASSIFIER_API_KEY) headers.Authorization = `Bearer ${process.env.CLASSIFIER_API_KEY}`;
    const response = await fetch(`${base}/predict`, {
      method: 'POST', headers, body: JSON.stringify({ text: checked.text }),
      signal: AbortSignal.timeout(25_000), cache: 'no-store'
    });
    if (!response.ok) {
      return NextResponse.json({ error: 'The classifier service could not process this passage. Check that the model API is online.' }, { status: 502 });
    }
    const data: unknown = await response.json();
    if (!data || typeof data !== 'object') throw new Error('Invalid response');
    const result = data as { label?: unknown; scores?: unknown };
    if (typeof result.label !== 'string' || !result.label.trim() || !validScores(result.scores)) {
      return NextResponse.json({ error: 'The classifier returned an unexpected response format.' }, { status: 502 });
    }
    const normalized = [...result.scores].sort((a, b) => b.score - a.score).slice(0, 5);
    const output: ClassificationResult = { configured: true, mode: 'live', label: result.label, scores: normalized };
    return NextResponse.json(output, { headers: { 'Cache-Control': 'no-store' } });
  } catch {
    return NextResponse.json({ error: 'Could not reach the classifier service. Please try again later.' }, { status: 502 });
  }
}
