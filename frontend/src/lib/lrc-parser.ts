/**
 * lrc-parser.ts — Parse .lrc (LyRiCs) files into structured time-stamped lines.
 *
 * LRC format:
 *   [mm:ss.xx] lyric text
 *   [ti:Song Title]
 *   [ar:Artist Name]
 *   [al:Album Name]
 *
 * A single line can have multiple timestamps (same lyric repeated at different times).
 */

export interface LrcLine {
  startTime: number;  // seconds
  endTime: number;    // seconds (next line's startTime, or Infinity for last line)
  text: string;
}

export interface LrcMetadata {
  title?: string;
  artist?: string;
  album?: string;
}

export interface ParsedLrc {
  metadata: LrcMetadata;
  lines: LrcLine[];
}

const TIME_TAG = /\[(\d{1,2}):(\d{2})\.(\d{2,3})\]/g;
const META_TAG = /^\[(ti|ar|al|length|by):(.+)\]$/i;

function parseSeconds(mm: string, ss: string, xx: string): number {
  return (
    parseInt(mm, 10) * 60 +
    parseInt(ss, 10) +
    parseInt(xx.padEnd(3, "0"), 10) / 1000
  );
}

export function parseLrc(content: string): ParsedLrc {
  const metadata: LrcMetadata = {};
  const raw: { time: number; text: string }[] = [];

  for (const rawLine of content.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line) continue;

    // Metadata tag?
    const meta = META_TAG.exec(line);
    if (meta) {
      const key = meta[1].toLowerCase();
      const value = meta[2].trim();
      if (key === "ti") metadata.title = value;
      else if (key === "ar") metadata.artist = value;
      else if (key === "al") metadata.album = value;
      continue;
    }

    // Collect all time tags, then the text is what remains
    const times: number[] = [];
    const text = line
      .replace(TIME_TAG, (_, mm, ss, xx) => {
        times.push(parseSeconds(mm, ss, xx));
        return "";
      })
      .trim();

    if (!text || !times.length) continue;

    // Skip instrumental/chorus markers like [*] or [Chorus]
    if (/^\[.*\]$/.test(text)) continue;

    for (const t of times) {
      raw.push({ time: t, text });
    }
  }

  // Sort chronologically
  raw.sort((a, b) => a.time - b.time);

  // Build LrcLines with endTime from the next line's startTime
  const lines: LrcLine[] = raw.map((item, i) => ({
    startTime: item.time,
    endTime: raw[i + 1]?.time ?? Infinity,
    text: item.text,
  }));

  return { metadata, lines };
}

/** Return the index of the currently active lyric line for a given playback time. */
export function findCurrentLineIndex(lines: LrcLine[], currentTime: number): number {
  let idx = -1;
  for (let i = 0; i < lines.length; i++) {
    if (currentTime >= lines[i].startTime) idx = i;
    else break;
  }
  return idx;
}
