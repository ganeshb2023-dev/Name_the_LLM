export type Score = { label: string; score: number };
export type ClassificationResult = {
  configured: boolean;
  label?: string;
  scores?: Score[];
  message?: string;
  mode?: 'live' | 'not-configured';
};
