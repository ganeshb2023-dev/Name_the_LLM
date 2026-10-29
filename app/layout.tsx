import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Name That LLM — AI Text Origin Analyzer',
  description: 'Analyze text with a transparent workflow for identifying the language model that generated it.',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
