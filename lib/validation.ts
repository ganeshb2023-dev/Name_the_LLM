export const MAX_TEXT_LENGTH = 20_000;
export function validatePassage(value: unknown): { ok: true; text: string } | { ok: false; message: string } {
  if (typeof value !== 'string') return { ok: false, message: 'Enter a text passage to analyze.' };
  const text = value.trim();
  if (text.length < 20) return { ok: false, message: 'Please enter at least 20 characters for a useful analysis.' };
  if (text.length > MAX_TEXT_LENGTH) return { ok: false, message: `Please keep the passage under ${MAX_TEXT_LENGTH.toLocaleString()} characters.` };
  return { ok: true, text };
}
