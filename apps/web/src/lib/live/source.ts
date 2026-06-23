import type { LiveSource, LiveSourceType } from "../types";

export function sourcePlaceholder(type: LiveSourceType): string {
  switch (type) {
    case "file":
      return "data/raw/clip.mp4";
    case "youtube":
      return "https://www.youtube.com/watch?v=...";
    case "obs":
      return "/dev/video2";
  }
}

export type SourceFormValues = {
  type: LiveSourceType;
  value: string;
  fileStart: string;
  fileEnd: string;
  youtubeStart: string;
  youtubeEnd: string;
  maxFps: number;
};

export function buildLiveSource(values: SourceFormValues): LiveSource | null {
  const base = { max_fps: values.maxFps };

  switch (values.type) {
    case "file":
      return {
        type: "file",
        path: values.value,
        start: values.fileStart,
        end: values.fileEnd,
        ...base,
      };
    case "youtube":
      return {
        type: "youtube",
        url: values.value,
        start: values.youtubeStart,
        end: values.youtubeEnd,
        ...base,
      };
    case "obs":
      return { type: "obs", mode: "device", device: values.value, ...base };
    default:
      return null;
  }
}
