export type Device = "cuda" | "cpu" | "mps";

export type LiveSourceType = "file" | "youtube" | "obs";

export type StateReadouts = {
  shot_prob: number;
  xg: number;
  turnover_prob: number;
  top_receiver_slot: number;
  top_receiver_prob: number;
  pass_end_x: number;
  pass_end_y: number;
};

export type LiveMetadata = {
  frame: number;
  latency_ms: number;
  width: number;
  height: number;
  classes: Record<string, number>;
  detections: number;
  possession: { class_name: string; team_id: number } | null;
  pressure: {
    pressure_side: string;
    pressure_score: number;
    pitch_territory_delta?: number | null;
  };
  state: StateReadouts | null;
  team_ready: boolean;
};

export type BaseLiveSource = {
  max_fps: number;
};

export type FileLiveSource = BaseLiveSource & {
  type: "file";
  path: string;
  start: string;
  end: string;
};

export type YoutubeLiveSource = BaseLiveSource & {
  type: "youtube";
  url: string;
  start: string;
  end: string;
};

export type ObsLiveSource = BaseLiveSource & {
  type: "obs";
  mode: "device";
  device: string;
};

export type LiveSource = FileLiveSource | YoutubeLiveSource | ObsLiveSource;

export type WsUrlConfig = string | undefined;
