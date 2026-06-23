<script lang="ts">
	import { onMount } from 'svelte';

	type StateReadouts = {
		shot_prob: number;
		xg: number;
		turnover_prob: number;
		top_receiver_slot: number;
		top_receiver_prob: number;
		pass_end_x: number;
		pass_end_y: number;
	};

	type LiveMetadata = {
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

	let device = $state('cuda');

	let liveSourceType = $state<'file' | 'youtube' | 'obs'>('file');
	let liveSourceValue = $state('');
	let liveFileStart = $state('00:00:00');
	let liveFileEnd = $state('00:02:00');
	let liveYoutubeStart = $state('00:00:00');
	let liveYoutubeEnd = $state('00:02:00');
	let liveMaxFps = $state(5);
	let liveRunning = $state(false);
	let liveError = $state('');
	let liveMeta = $state<LiveMetadata | null>(null);
	let ws: WebSocket | null = null;
	let wsReadyPromise: Promise<WebSocket> | null = null;
	let outputCanvas = $state<HTMLCanvasElement | null>(null);
	let outputCtx = $state<CanvasRenderingContext2D | null>(null);
	let objectUrl: string | null = null;

	const wsUrl = () => {
		const cfg = (window as any).__FOOTBALLAI_WS__ as string | undefined;
		if (cfg) return cfg;
		const loc = window.location;
		const protocol = loc.protocol === 'https:' ? 'wss:' : 'ws:';
		return `${protocol}//${loc.host}/ws`;
	};

	function ensureOutputCtx() {
		if (!outputCtx && outputCanvas) {
			outputCtx = outputCanvas.getContext('2d');
		}
		return outputCtx;
	}

	function sourcePlaceholder() {
		switch (liveSourceType) {
			case 'file':
				return 'data/raw/clip.mp4';
			case 'youtube':
				return 'https://www.youtube.com/watch?v=...';
			case 'obs':
				return '/dev/video2';
		}
	}

	function buildLiveSource() {
		const base = { max_fps: liveMaxFps };
		switch (liveSourceType) {
			case 'file':
				return {
					type: 'file',
					path: liveSourceValue,
					start: liveFileStart,
					end: liveFileEnd,
					...base,
				};
			case 'youtube':
				return {
					type: 'youtube',
					url: liveSourceValue,
					start: liveYoutubeStart,
					end: liveYoutubeEnd,
					...base,
				};
			case 'obs': {
				return { type: 'obs', mode: 'device', device: liveSourceValue, ...base };
			}
		}
	}

	function handleWsMessage(event: MessageEvent<string | Blob>) {
		if (typeof event.data === 'string') {
			try {
				const payload = JSON.parse(event.data) as { type: string } & Record<string, unknown>;
				if (payload.type === 'metadata') {
					liveMeta = payload as unknown as LiveMetadata;
				} else if (payload.type === 'error') {
					liveError = (payload.message as string) ?? 'Server error';
				}
			} catch {
				// ignore non-JSON text
			}
		} else if (event.data instanceof Blob) {
			if (objectUrl) URL.revokeObjectURL(objectUrl);
			objectUrl = URL.createObjectURL(event.data);
			const ctx = ensureOutputCtx();
			const canvas = outputCanvas;
			if (ctx && canvas) {
				const img = new Image();
				img.onload = () => {
					canvas.width = img.naturalWidth;
					canvas.height = img.naturalHeight;
					ctx.drawImage(img, 0, 0);
				};
				img.src = objectUrl;
			}
		}
	}

	function openWsIfNeeded(): Promise<WebSocket> {
		if (ws?.readyState === WebSocket.OPEN) return Promise.resolve(ws);
		if (wsReadyPromise) return wsReadyPromise;

		wsReadyPromise = new Promise<WebSocket>((resolve, reject) => {
			if (ws && ws.readyState !== WebSocket.CLOSED && ws.readyState !== WebSocket.CLOSING) {
				try {
					ws.close();
				} catch {
					// ignore
				}
			}

			const socket = new WebSocket(wsUrl());
			ws = socket;
			socket.binaryType = 'blob';
			socket.onmessage = handleWsMessage;
			socket.onopen = () => resolve(socket);
			socket.onerror = () => reject(new Error('WebSocket error. Is the Python server running?'));
			socket.onclose = () => {
				if (ws === socket) {
					ws = null;
					liveRunning = false;
				}
			};
		}).finally(() => {
			wsReadyPromise = null;
		});

		return wsReadyPromise;
	}

	async function startLive() {
		liveError = '';
		liveMeta = null;

		if (!liveSourceValue.trim()) {
			liveError = 'Please enter a source value';
			return;
		}

		const source = buildLiveSource();
		if (!source) {
			liveError = 'Invalid source configuration';
			return;
		}

		liveRunning = true;
		try {
			const socket = await openWsIfNeeded();
			socket.send(JSON.stringify({ action: 'configure', options: { device } }));
			socket.send(JSON.stringify({ action: 'live_start', source, options: { device } }));
		} catch (err) {
			liveRunning = false;
			liveError = err instanceof Error ? err.message : 'Failed to start live stream';
		}
	}

	function stopLive() {
		liveRunning = false;
		try {
			ws?.send(JSON.stringify({ action: 'live_stop' }));
		} catch {
			// ignore
		}
		ws?.close();
		ws = null;
		wsReadyPromise = null;
		if (objectUrl) {
			URL.revokeObjectURL(objectUrl);
			objectUrl = null;
		}
	}

	onMount(() => {
		ensureOutputCtx();
		return () => stopLive();
	});
</script>

<main class="shell">
	<aside class="left-pane">
		<header class="topbar">
			<h1>FootballAI</h1>
		</header>

		<section class="panel live-form">
			<p class="eyebrow">Live Stream</p>
			<h2>Send a source to the backend and get an annotated stream back</h2>
			<p class="muted">
				The backend owns decoding. Paste a local file path, a YouTube link, or an OBS device path.
				Annotated frames are pushed back over the WebSocket.
			</p>
			<div class="form-grid">
				<label class="field">
					<span>Source type</span>
					<select bind:value={liveSourceType}>
						<option value="file">Local MP4 file</option>
						<option value="youtube">YouTube URL</option>
						<option value="obs">OBS</option>
					</select>
				</label>
				<label class="field span-2">
					<span>Source</span>
					<input type="text" bind:value={liveSourceValue} placeholder={sourcePlaceholder()} />
				</label>
				{#if liveSourceType === 'file'}
					<label class="field">
						<span>Start</span>
						<input type="text" bind:value={liveFileStart} placeholder="00:00:00" />
					</label>
					<label class="field">
						<span>End</span>
						<input type="text" bind:value={liveFileEnd} placeholder="00:02:00" />
					</label>
				{/if}
				{#if liveSourceType === 'youtube'}
					<label class="field">
						<span>Start</span>
						<input type="text" bind:value={liveYoutubeStart} placeholder="00:00:00" />
					</label>
					<label class="field">
						<span>End</span>
						<input type="text" bind:value={liveYoutubeEnd} placeholder="00:02:00" />
					</label>
				{/if}
				<label class="field">
					<span>Max FPS</span>
					<input type="number" bind:value={liveMaxFps} min="1" max="60" />
				</label>
				<label class="field">
					<span>Device</span>
					<select bind:value={device}>
						<option value="cuda">cuda</option>
						<option value="cpu">cpu</option>
						<option value="mps">mps</option>
					</select>
				</label>
			</div>
			<div class="actions">
				{#if liveRunning}
					<button type="button" class="secondary" onclick={stopLive}>Stop live</button>
				{:else}
					<button type="button" class="primary" onclick={startLive}>Start live</button>
				{/if}
			</div>
			{#if liveError}
				<p class="error">{liveError}</p>
			{/if}
		</section>
	</aside>

	<section class="right-pane">
		<section class="live-metrics">
			<div>
				<span class="metric-value">{liveMeta?.state ? `${Math.round(liveMeta.state.shot_prob * 100)}%` : '-'}</span>
				<span class="metric-label">shot prob</span>
			</div>
			<div>
				<span class="metric-value">{liveMeta?.state ? liveMeta.state.xg.toFixed(2) : '-'}</span>
				<span class="metric-label">goal threat</span>
			</div>
			<div>
				<span class="metric-value">{liveMeta?.state ? `${Math.round(liveMeta.state.turnover_prob * 100)}%` : '-'}</span>
				<span class="metric-label">turnover prob</span>
			</div>
			<div>
				<span class="metric-value">{liveMeta?.state ? `${liveMeta.state.top_receiver_slot} (${Math.round(liveMeta.state.top_receiver_prob * 100)}%)` : '-'}</span>
				<span class="metric-label">pass receiver</span>
			</div>
			<div>
				<span class="metric-value">{liveMeta?.latency_ms ? `${Math.round(liveMeta.latency_ms)}ms` : '-'}</span>
				<span class="metric-label">latency</span>
			</div>
		</section>

		<section class="live-stage">
			<div class="video-frame output-frame">
				<canvas bind:this={outputCanvas} class="output-canvas"></canvas>
				{#if !liveMeta}
					<div class="placeholder">Annotated stream will appear here</div>
				{/if}
			</div>
		</section>
	</section>
</main>

<style>
	:global(body) {
		margin: 0;
		background: #0f1214;
		color: #f3f5f4;
		font-family:
			Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
	}

	.shell {
		display: grid;
		grid-template-columns: 320px minmax(0, 1fr);
		gap: 24px;
		height: 100vh;
		overflow: hidden;
		padding: 24px;
		box-sizing: border-box;
	}

	.left-pane {
		display: flex;
		flex-direction: column;
		gap: 18px;
		overflow-y: auto;
		min-width: 0;
	}

	.right-pane {
		display: flex;
		flex-direction: column;
		gap: 18px;
		min-width: 0;
		overflow: hidden;
	}

	.topbar {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 20px;
	}

	.eyebrow {
		margin: 0 0 4px;
		color: #8fb8a0;
		font-size: 13px;
		font-weight: 700;
		text-transform: uppercase;
	}

	h1,
	h2,
	p {
		margin: 0;
	}

	h1 {
		font-size: 32px;
		line-height: 1.1;
	}

	h2 {
		margin-bottom: 14px;
		font-size: 16px;
	}

	button,
	a {
		border: 1px solid #3b4541;
		background: #1c2421;
		color: #f3f5f4;
		border-radius: 6px;
		padding: 10px 12px;
		font: inherit;
		text-decoration: none;
		cursor: pointer;
	}

	button:hover,
	a:hover {
		border-color: #77c996;
	}

	button.primary {
		background: #2a3b33;
		border-color: #77c996;
		color: #f3f5f4;
		font-weight: 600;
	}

	button.secondary {
		background: #161b1a;
	}

	.panel {
		border: 1px solid #26302c;
		background: #161b1a;
		border-radius: 8px;
		padding: 16px;
	}

	.live-form .form-grid {
		display: grid;
		grid-template-columns: 1fr;
		gap: 12px;
		margin: 14px 0;
	}

	.field {
		display: flex;
		flex-direction: column;
		gap: 6px;
		font-size: 13px;
	}

	.field span {
		color: #aab4af;
	}

	.field input,
	.field select {
		background: #0d1110;
		border: 1px solid #3b4541;
		border-radius: 6px;
		padding: 8px 10px;
		color: #f3f5f4;
		font: inherit;
	}

	.field select {
		padding: 8px 28px 8px 10px;
		appearance: none;
		background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%23aab4af' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m6 9 6 6 6-6'/%3E%3C/svg%3E");
		background-repeat: no-repeat;
		background-position: right 10px center;
	}

	.field.span-2 {
		grid-column: auto;
	}

	.field.checkbox {
		flex-direction: row;
		align-items: center;
		gap: 8px;
	}

	.muted {
		color: #aab4af;
		line-height: 1.4;
		margin: 8px 0 12px;
	}

	.actions {
		display: flex;
		gap: 10px;
		margin-top: 12px;
	}

	.live-stage {
		display: flex;
		flex-direction: column;
		gap: 14px;
		min-width: 0;
		flex: 1 1 auto;
		min-height: 0;
	}

	.video-frame {
		background: #050706;
		border: 1px solid #26302c;
		border-radius: 8px;
		overflow: hidden;
		flex: 1 1 auto;
		position: relative;
		min-height: 0;
	}

	.output-frame {
		min-height: 0;
	}

	.output-canvas {
		width: 100%;
		height: 100%;
		display: block;
		object-fit: contain;
	}

	.placeholder {
		position: absolute;
		inset: 0;
		display: grid;
		place-items: center;
		color: #aab4af;
	}

	.live-metrics > div,
	.empty {
		border: 1px solid #26302c;
		background: #161b1a;
		border-radius: 8px;
	}

	.live-metrics > div {
		display: grid;
		gap: 4px;
		padding: 14px;
	}

	.metric-value {
		font-size: 24px;
		font-weight: 800;
	}

	.metric-label {
		color: #aab4af;
		font-size: 13px;
	}

	.live-metrics {
		display: grid;
		grid-template-columns: repeat(5, minmax(0, 1fr));
		gap: 10px;
	}

	.empty {
		padding: 24px;
		color: #c4ccc8;
	}

	.error {
		color: #ffc2b8;
	}

	@media (max-width: 900px) {
		.shell {
			grid-template-columns: 1fr;
			height: auto;
			overflow: visible;
			padding: 16px;
		}

		.left-pane,
		.right-pane {
			overflow: visible;
		}

		.live-metrics {
			grid-template-columns: 1fr;
		}
	}
</style>
