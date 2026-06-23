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
	let theme = $state<'light' | 'dark'>(
		typeof window !== 'undefined' && window.matchMedia('(prefers-color-scheme: dark)').matches
			? 'dark'
			: 'light'
	);

	$effect(() => {
		document.documentElement.setAttribute('data-theme', theme);
	});
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

	function getInitialTheme(): 'light' | 'dark' {
		if (typeof window === 'undefined') return 'light';
		if (window.matchMedia('(prefers-color-scheme: dark)').matches) return 'dark';
		return 'light';
	}

	onMount(() => {
		theme = getInitialTheme();
		ensureOutputCtx();
		return () => stopLive();
	});
</script>

<main class="shell">
	<aside class="left-pane">
		<header class="topbar">
			<h1>FootballAI</h1>
			<div class="topbar-tools">
				<button
					type="button"
					class="theme-toggle"
					aria-label={theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}
					onclick={() => (theme = theme === 'light' ? 'dark' : 'light')}
				>
					<span class="theme-icon" aria-hidden="true">{theme === 'light' ? '☀️' : '🌙'}</span>
					<span class="theme-label">{theme === 'light' ? 'Light' : 'Dark'}</span>
				</button>
			</div>
		</header>

		<section class="panel live-form">
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
				<span class="metric-value">{liveMeta?.state ? `${Math.round(liveMeta.state.shot_prob * 100)}%` : '—'}</span>
				<span class="metric-label">shot chance</span>
			</div>
			<div>
				<span class="metric-value">{liveMeta?.state ? liveMeta.state.xg.toFixed(2) : '—'}</span>
				<span class="metric-label">goal threat</span>
			</div>
			<div>
				<span class="metric-value">{liveMeta?.state ? `${Math.round(liveMeta.state.turnover_prob * 100)}%` : '—'}</span>
				<span class="metric-label">turnover risk</span>
			</div>
			<div>
				<span class="metric-value">{liveMeta?.state ? `${liveMeta.state.top_receiver_slot} (${Math.round(liveMeta.state.top_receiver_prob * 100)}%)` : '—'}</span>
				<span class="metric-label">receiver (pass prob)</span>
			</div>
			<div>
				<span class="metric-value">{liveMeta?.latency_ms ? `${Math.round(liveMeta.latency_ms)}ms` : '—'}</span>
				<span class="metric-label">latency</span>
			</div>
		</section>

		<section class="live-stage">
			<div class="video-frame output-frame">
				<canvas bind:this={outputCanvas} class="output-canvas"></canvas>
				{#if !liveMeta}
					<div class="placeholder">
						<span>Annotated stream will appear here</span>
					</div>
				{/if}
			</div>
		</section>
	</section>
</main>

<style>
	:global(body) {
		margin: 0;
		background: var(--paper);
		color: var(--ink);
		font-family: var(--font-ui);
		font-size: 13px;
		line-height: 1.5;
	}

	.shell {
		display: grid;
		grid-template-columns: 340px minmax(0, 1fr);
		gap: var(--space-5);
		height: 100vh;
		overflow: hidden;
		padding: var(--space-5);
		box-sizing: border-box;
	}

	.left-pane {
		display: flex;
		flex-direction: column;
		gap: var(--space-5);
		overflow-y: auto;
		min-width: 0;
	}

	.right-pane {
		display: flex;
		flex-direction: column;
		gap: var(--space-4);
		min-width: 0;
		overflow: hidden;
	}

	.topbar {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--space-4);
		min-height: 36px;
	}

	.topbar h1 {
		font-family: var(--font-brand);
		font-size: 26px;
		font-weight: 700;
		letter-spacing: -0.03em;
		color: var(--ink);
		line-height: 1;
	}

	.topbar-tools {
		display: flex;
		align-items: center;
		gap: var(--space-2);
	}

	.theme-toggle {
		display: inline-flex;
		align-items: center;
		gap: var(--space-1);
		padding: 5px 10px;
		border: 1px solid var(--line);
		background: var(--surface);
		color: var(--ink);
		border-radius: 999px;
		font: inherit;
		font-size: 12px;
		font-weight: 500;
		line-height: 1;
		cursor: pointer;
		transition: border-color 0.15s ease, background 0.15s ease;
	}

	.theme-toggle:hover {
		border-color: var(--signal);
	}

	.theme-icon {
		font-size: 13px;
		line-height: 1;
	}

	.theme-label {
		font-size: 11px;
		line-height: 1;
	}

	h1,
	h2,
	p {
		margin: 0;
	}

	h2 {
		font-family: var(--font-ui);
		font-size: 15px;
		font-weight: 500;
		line-height: 1.35;
		margin-bottom: var(--space-3);
	}

	button {
		border: 1px solid var(--line-strong);
		background: var(--surface);
		color: var(--ink);
		border-radius: var(--radius-sm);
		padding: 10px 14px;
		font: inherit;
		font-weight: 500;
		cursor: pointer;
		transition: border-color 0.15s ease, background 0.15s ease;
	}

	button:hover {
		border-color: var(--signal);
	}

	button.primary {
		background: var(--signal);
		border-color: var(--signal);
		color: #fff;
		font-weight: 600;
	}

	button.primary:hover {
		background: var(--signal-hover);
		border-color: var(--signal-hover);
	}

	button.secondary {
		background: var(--surface);
	}

	.panel {
		border: 1px solid var(--line);
		background: var(--surface);
		border-radius: var(--radius);
		padding: var(--space-4);
	}

	.live-form .form-grid {
		display: grid;
		grid-template-columns: 1fr;
		gap: var(--space-3);
		margin: var(--space-3) 0;
	}

	.field {
		display: flex;
		flex-direction: column;
		gap: var(--space-1);
		font-size: 13px;
	}

	.field span {
		color: var(--graphite);
		font-weight: 500;
	}

	.field input,
	.field select {
		background: var(--paper);
		border: 1px solid var(--line);
		border-radius: var(--radius-sm);
		padding: 9px 10px;
		color: var(--ink);
		font: inherit;
		transition: border-color 0.15s ease;
	}

	.field input:focus,
	.field select:focus {
		border-color: var(--signal);
	}

	.field select {
		padding: 9px 28px 9px 10px;
		appearance: none;
		background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%236e6e6e' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m6 9 6 6 6-6'/%3E%3C/svg%3E");
		background-repeat: no-repeat;
		background-position: right 10px center;
	}

	.field.span-2 {
		grid-column: auto;
	}

	.muted {
		color: var(--graphite);
		line-height: 1.5;
		margin: 0 0 var(--space-3);
	}

	.actions {
		display: flex;
		gap: var(--space-2);
		margin-top: var(--space-3);
	}

	.live-stage {
		display: flex;
		flex-direction: column;
		gap: var(--space-4);
		min-width: 0;
		flex: 1 1 auto;
		min-height: 0;
	}

	.video-frame {
		background: var(--stage-bg);
		border: 1px solid var(--line);
		border-radius: var(--radius);
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
		color: var(--graphite);
		font-size: 13px;
		padding: var(--space-4);
		text-align: center;
	}

	.placeholder span {
		background: var(--stage-paper);
		border: 1px dashed var(--line-strong);
		border-radius: var(--radius);
		padding: var(--space-4) var(--space-5);
		box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
	}

	.live-metrics {
		display: grid;
		grid-template-columns: repeat(5, minmax(0, 1fr));
		gap: var(--space-3);
	}

	.live-metrics > div {
		border: 1px solid var(--line);
		background: var(--surface);
		border-radius: var(--radius);
		display: grid;
		gap: var(--space-1);
		padding: var(--space-4);
	}

	.metric-value {
		font-family: var(--font-brand);
		font-size: 24px;
		font-weight: 700;
		letter-spacing: -0.02em;
		font-variant-numeric: tabular-nums;
		line-height: 1.2;
	}

	.metric-label {
		color: var(--graphite);
		font-size: 10px;
		font-weight: 600;
		letter-spacing: 0.06em;
		text-transform: uppercase;
	}

	.error {
		color: var(--error);
		margin-top: var(--space-3);
	}

	@media (max-width: 980px) {
		.shell {
			grid-template-columns: 1fr;
			height: auto;
			overflow: visible;
			padding: var(--space-4);
		}

		.left-pane,
		.right-pane {
			overflow: visible;
		}

		.live-metrics {
			grid-template-columns: repeat(2, minmax(0, 1fr));
		}

		.video-frame {
			aspect-ratio: 16 / 9;
			min-height: 280px;
		}
	}

	@media (max-width: 560px) {
		.live-metrics {
			grid-template-columns: 1fr;
		}
	}
</style>
