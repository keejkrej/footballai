<script lang="ts">
	import { onMount } from 'svelte';
	import Hls from 'hls.js';

	type RunSummary = {
		id: string;
		label: string;
		video: string;
		csv: string | null;
		sizeBytes: number;
		durationLabel: string;
		detections: number;
		classes: Record<string, number>;
	};

	type Job = {
		id: string;
		status: string;
		progress: number;
		message: string;
		videoUrl?: string;
		csvUrl?: string | null;
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
		team_ready: boolean;
	};

	let mode = $state<'full' | 'live'>('full');
	let runs = $state<RunSummary[]>([]);
	let selectedId = $state('');
	let loading = $state(true);
	let error = $state('');

	// Full mode form
	let youtubeUrl = $state('');
	let startTime = $state('00:00:00');
	let endTime = $state('00:02:00');
	let device = $state('cuda');
	let maxFrames = $state(0);
	let stride = $state(1);
	let skipTeamFit = $state(false);
	let runningJob = $state<Job | null>(null);
	let showProgress = $state(false);
	let pollInterval: ReturnType<typeof setInterval> | null = null;

	// Live mode state
	let streamUrl = $state('');
	let captureFps = $state(5);
	let jpegQuality = $state(0.7);
	let liveRunning = $state(false);
	let liveError = $state('');
	let liveMeta = $state<LiveMetadata | null>(null);
	let ws: WebSocket | null = null;
	let captureTimer: ReturnType<typeof setInterval> | null = null;
	let videoEl = $state<HTMLVideoElement | null>(null);
	let captureCanvas = $state<HTMLCanvasElement | null>(null);
	let outputCanvas = $state<HTMLCanvasElement | null>(null);
	let outputCtx = $state<CanvasRenderingContext2D | null>(null);
	let objectUrl: string | null = null;
	let hls: Hls | null = null;

	const selectedRun = $derived(runs.find((run) => run.id === selectedId) ?? runs[0]);
	const totalDetections = $derived(selectedRun?.detections ?? 0);
	const classEntries = $derived(Object.entries(selectedRun?.classes ?? {}).sort((a, b) => b[1] - a[1]));

	function formatBytes(bytes: number) {
		if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
		return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
	}

	async function loadRuns() {
		loading = true;
		error = '';
		try {
			const response = await fetch('/api/runs');
			if (!response.ok) throw new Error(`Failed to load runs: ${response.status}`);
			const data = (await response.json()) as { runs: RunSummary[] };
			runs = data.runs;
			selectedId = data.runs[0]?.id ?? '';
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to load runs';
		} finally {
			loading = false;
		}
	}

	async function startFullPipeline() {
		if (!youtubeUrl.trim()) {
			error = 'Please enter a YouTube URL';
			return;
		}
		error = '';
		showProgress = true;
		try {
			const response = await fetch('/api/full', {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: JSON.stringify({
					youtubeUrl,
					start: startTime,
					end: endTime,
					device,
					maxFrames,
					stride,
					skipTeamFit
				})
			});
			const data = (await response.json()) as { jobId?: string; error?: string };
			if (!response.ok || data.error) {
				throw new Error(data.error || `Failed to start pipeline: ${response.status}`);
			}
			const jobId = data.jobId!;
			runningJob = { id: jobId, status: 'pending', progress: 0, message: 'Queued' };
			pollInterval = setInterval(async () => {
				const res = await fetch(`/api/jobs/${jobId}`);
				if (!res.ok) return;
				const payload = (await res.json()) as { job: Job };
				runningJob = payload.job;
				if (payload.job.status === 'done' || payload.job.status === 'error') {
					if (pollInterval) clearInterval(pollInterval);
					pollInterval = null;
					if (payload.job.status === 'done') {
						await loadRuns();
						const run = runs.find((r) => r.video === payload.job.videoUrl);
						if (run) selectedId = run.id;
					}
					setTimeout(() => {
						showProgress = false;
						runningJob = null;
					}, 2000);
				}
			}, 1000);
		} catch (err) {
			showProgress = false;
			runningJob = null;
			error = err instanceof Error ? err.message : 'Failed to start pipeline';
		}
	}

	function stopFullPipeline() {
		if (pollInterval) clearInterval(pollInterval);
		pollInterval = null;
		showProgress = false;
		runningJob = null;
	}

	function connectLive() {
		liveError = '';
		liveMeta = null;
		ws = new WebSocket('ws://localhost:8000');
		ws.binaryType = 'blob';

		ws.onopen = () => {
			ws?.send(JSON.stringify({ action: 'configure', options: { device: 'cuda' } }));
			startCapture();
		};

		ws.onmessage = (event) => {
			if (typeof event.data === 'string') {
				try {
					const payload = JSON.parse(event.data) as { type: string } & Partial<LiveMetadata>;
					if (payload.type === 'metadata') {
						liveMeta = payload as unknown as LiveMetadata;
					}
				} catch {
					// ignore non-JSON text
				}
			} else if (event.data instanceof Blob) {
				if (objectUrl) URL.revokeObjectURL(objectUrl);
				objectUrl = URL.createObjectURL(event.data);
				const ctx = outputCtx;
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
		};

		ws.onerror = () => {
			liveError = 'WebSocket error. Is `uv run inference live` running on port 8000?';
			stopLive();
		};

		ws.onclose = () => {
			stopLive();
		};
	}

	async function startLive() {
		if (!streamUrl.trim()) {
			liveError = 'Please enter a stream URL or 0/webcam';
			return;
		}
		liveRunning = true;
		liveError = '';

		const url = streamUrl.trim();
		const isWebcam = url === '0' || url.toLowerCase() === 'webcam';

		if (isWebcam) {
			try {
				const mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
				if (videoEl) videoEl.srcObject = mediaStream;
			} catch (err) {
				liveError = `Webcam failed: ${err instanceof Error ? err.message : String(err)}`;
				stopLive();
				return;
			}
		} else if (Hls.isSupported() && (url.endsWith('.m3u8') || url.includes('m3u8'))) {
			hls = new Hls();
			if (videoEl) {
				hls.loadSource(url);
				hls.attachMedia(videoEl);
				hls.on(Hls.Events.MANIFEST_PARSED, () => videoEl?.play());
				hls.on(Hls.Events.ERROR, (_, data) => {
					if (data.fatal) {
						liveError = `HLS error: ${data.type}`;
						stopLive();
					}
				});
			}
		} else {
			if (videoEl) {
				videoEl.src = url;
				videoEl.crossOrigin = 'anonymous';
				videoEl.play().catch((err) => {
					liveError = `Video playback failed: ${err instanceof Error ? err.message : String(err)}`;
					stopLive();
				});
			}
		}

		videoEl?.addEventListener('playing', connectLive, { once: true });
	}

	function startCapture() {
		if (!videoEl || !captureCanvas) return;
		const intervalMs = Math.max(50, 1000 / captureFps);
		captureTimer = setInterval(() => {
			if (!videoEl || !captureCanvas || !ws || ws.readyState !== WebSocket.OPEN) return;
			const ctx = captureCanvas.getContext('2d');
			if (!ctx) return;
			captureCanvas.width = videoEl.videoWidth || 640;
			captureCanvas.height = videoEl.videoHeight || 360;
			ctx.drawImage(videoEl, 0, 0, captureCanvas.width, captureCanvas.height);
			captureCanvas.toBlob(
				(blob) => {
					if (blob && ws?.readyState === WebSocket.OPEN) {
						ws.send(blob);
					}
				},
				'image/jpeg',
				jpegQuality
			);
		}, intervalMs);
	}

	function stopLive() {
		liveRunning = false;
		if (captureTimer) clearInterval(captureTimer);
		captureTimer = null;
		if (ws) {
			try {
				ws.send(JSON.stringify({ action: 'stop' }));
			} catch {
				// ignore
			}
			ws.close();
		}
		ws = null;
		if (hls) {
			hls.destroy();
			hls = null;
		}
		if (videoEl) {
			videoEl.pause();
			videoEl.src = '';
			videoEl.srcObject = null;
		}
		if (objectUrl) {
			URL.revokeObjectURL(objectUrl);
			objectUrl = null;
		}
	}

	onMount(() => {
		loadRuns();
		const canvas = outputCanvas;
		if (canvas) {
			outputCtx = canvas.getContext('2d');
		}
		return () => {
			stopFullPipeline();
			stopLive();
		};
	});
</script>

<svelte:head>
	<title>FootballAI Overlay Lab</title>
	<meta
		name="description"
		content="Run football player detection overlays on YouTube clips or live streams."
	/>
</svelte:head>

<main class="shell">
	<header class="topbar">
		<div>
			<p class="eyebrow">FootballAI</p>
			<h1>Overlay Lab</h1>
		</div>
		<div class="tabs">
			<button type="button" class:active={mode === 'full'} onclick={() => (mode = 'full')}>Full</button>
			<button type="button" class:active={mode === 'live'} onclick={() => (mode = 'live')}>Live</button>
		</div>
	</header>

	{#if mode === 'full'}
		<section class="panel full-form">
			<p class="eyebrow">Full Match</p>
			<h2>Analyze a YouTube clip</h2>
			<div class="form-grid">
				<label class="field span-2">
					<span>YouTube URL</span>
					<input type="text" bind:value={youtubeUrl} placeholder="https://www.youtube.com/watch?v=..." />
				</label>
				<label class="field">
					<span>Start</span>
					<input type="text" bind:value={startTime} placeholder="00:00:00" />
				</label>
				<label class="field">
					<span>End</span>
					<input type="text" bind:value={endTime} placeholder="00:02:00" />
				</label>
				<label class="field">
					<span>Device</span>
					<select bind:value={device}>
						<option value="cuda">cuda</option>
						<option value="cpu">cpu</option>
						<option value="mps">mps</option>
					</select>
				</label>
				<label class="field">
					<span>Max frames</span>
					<input type="number" bind:value={maxFrames} min="0" />
				</label>
				<label class="field">
					<span>Stride</span>
					<input type="number" bind:value={stride} min="1" />
				</label>
				<label class="field checkbox">
					<input type="checkbox" bind:checked={skipTeamFit} />
					<span>Skip team fit (faster)</span>
				</label>
			</div>
			<button type="button" class="primary" onclick={startFullPipeline}>Run pipeline</button>
		</section>

		{#if showProgress && runningJob}
			<div class="modal-backdrop">
				<div class="modal">
					<h3>Running pipeline</h3>
					<p class="status">{runningJob.message}</p>
					<div class="progress-track">
						<div class="progress-fill" style={`width: ${runningJob.progress}%`}></div>
					</div>
					<p class="muted">{runningJob.progress}% — {runningJob.status}</p>
					<button type="button" class="secondary" onclick={stopFullPipeline}>Close</button>
				</div>
			</div>
		{/if}

		{#if error}
			<section class="empty error">{error}</section>
		{/if}

		{#if loading}
			<section class="empty">Loading generated overlays...</section>
		{:else if runs.length === 0}
			<section class="empty">No overlay videos found. Run the Full pipeline above to create one.</section>
		{:else}
			<section class="workspace">
				<aside class="runlist" aria-label="Generated overlay runs">
					{#each runs as run}
						<button
							type="button"
							class:active={run.id === selectedRun?.id}
							onclick={() => (selectedId = run.id)}
						>
							<span>{run.label}</span>
							<small>{run.detections.toLocaleString()} detections · {formatBytes(run.sizeBytes)}</small>
						</button>
					{/each}
				</aside>

				<section class="review">
					<div class="video-frame">
						{#if selectedRun}
							<video src={selectedRun.video} controls playsinline aria-label={selectedRun.label}>
								<track kind="captions" />
							</video>
						{/if}
					</div>

					<div class="metrics">
						<div>
							<span class="metric-value">{totalDetections.toLocaleString()}</span>
							<span class="metric-label">detections</span>
						</div>
						<div>
							<span class="metric-value">{classEntries.length}</span>
							<span class="metric-label">classes</span>
						</div>
						<div>
							<span class="metric-value">{selectedRun ? formatBytes(selectedRun.sizeBytes) : '-'}</span>
							<span class="metric-label">video size</span>
						</div>
					</div>

					<div class="detail-grid">
						<section class="panel">
							<h2>Class Breakdown</h2>
							{#if classEntries.length === 0}
								<p>No class metadata available for this run.</p>
							{:else}
								<div class="bars">
									{#each classEntries as [className, count]}
										<div class="bar-row">
											<div class="bar-label">
												<span>{className}</span>
												<strong>{count.toLocaleString()}</strong>
											</div>
											<div class="bar-track">
												<div style={`width: ${(count / Math.max(1, totalDetections)) * 100}%`}></div>
											</div>
										</div>
									{/each}
								</div>
							{/if}
						</section>

						<section class="panel">
							<h2>Pipeline</h2>
							<code>
								uv run inference full --input {selectedRun?.video.replace('/media/', 'data/outputs/').replace('_overlay.mp4', '.mp4') ?? 'data/raw/clip.mp4'} --output {selectedRun?.video.replace('/media/', 'data/outputs/') ?? 'data/outputs/overlay.mp4'}
							</code>
							{#if selectedRun?.csv}
								<a href={selectedRun.csv}>Download CSV</a>
							{/if}
						</section>
					</div>
				</section>
			</section>
		{/if}
	{:else}
		<section class="panel live-form">
			<p class="eyebrow">Live Stream</p>
			<h2>Stream a source and get annotated frames back</h2>
			<p class="muted">
				Paste an HLS / DASH / MP4 stream URL, or type <code>0</code> or <code>webcam</code> for your camera.
				The browser captures frames and sends them to the Python WebSocket server on
				<code>ws://localhost:8000</code>.
			</p>
			<div class="form-grid">
				<label class="field span-2">
					<span>Stream URL</span>
					<input type="text" bind:value={streamUrl} placeholder="https://example.com/stream.m3u8" />
				</label>
				<label class="field">
					<span>Capture FPS</span>
					<input type="number" bind:value={captureFps} min="1" max="30" />
				</label>
				<label class="field">
					<span>JPEG quality</span>
					<input type="number" bind:value={jpegQuality} min="0.1" max="1" step="0.1" />
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

		<section class="live-stage">
			<video bind:this={videoEl} class="hidden-video" playsinline muted></video>
			<canvas bind:this={captureCanvas} class="hidden-canvas"></canvas>
			<div class="video-frame output-frame">
				<canvas bind:this={outputCanvas} class="output-canvas"></canvas>
				{#if !liveMeta}
					<div class="placeholder">Annotated stream will appear here</div>
				{/if}
			</div>

			{#if liveMeta}
				<section class="live-metrics">
					<div>
						<span class="metric-value">{liveMeta.latency_ms ? `${Math.round(liveMeta.latency_ms)}ms` : '-'}</span>
						<span class="metric-label">model latency</span>
					</div>
					<div>
						<span class="metric-value">{liveMeta.detections}</span>
						<span class="metric-label">detections</span>
					</div>
					<div>
						<span class="metric-value">{liveMeta.pressure?.pressure_side ?? '-'}</span>
						<span class="metric-label">pressure side</span>
					</div>
					<div>
						<span class="metric-value">{liveMeta.pressure?.pressure_score ?? 0}</span>
						<span class="metric-label">pressure score</span>
					</div>
					<div>
						<span class="metric-value">{liveMeta.team_ready ? 'ready' : 'warmup'}</span>
						<span class="metric-label">teams</span>
					</div>
				</section>
			{/if}
		</section>
	{/if}
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
		min-height: 100vh;
		padding: 24px;
		box-sizing: border-box;
	}

	.topbar {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 20px;
		margin-bottom: 20px;
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

	h3 {
		margin: 0 0 12px;
	}

	.tabs {
		display: flex;
		gap: 8px;
	}

	.tabs button {
		border: 1px solid #3b4541;
		background: #1c2421;
		color: #f3f5f4;
		border-radius: 6px;
		padding: 10px 16px;
		font: inherit;
		cursor: pointer;
	}

	.tabs button.active {
		border-color: #77c996;
		background: #213029;
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
		margin-bottom: 18px;
	}

	.full-form .form-grid,
	.live-form .form-grid {
		display: grid;
		grid-template-columns: repeat(4, minmax(0, 1fr));
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

	.field.span-2 {
		grid-column: span 2;
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

	.workspace {
		display: grid;
		grid-template-columns: 280px minmax(0, 1fr);
		gap: 18px;
	}

	.runlist {
		display: flex;
		flex-direction: column;
		gap: 8px;
	}

	.runlist button {
		display: grid;
		gap: 6px;
		text-align: left;
		background: #161b1a;
	}

	.runlist button.active {
		border-color: #77c996;
		background: #213029;
	}

	.runlist small {
		color: #aab4af;
		font-size: 12px;
	}

	.review {
		min-width: 0;
	}

	.video-frame {
		background: #050706;
		border: 1px solid #26302c;
		border-radius: 8px;
		overflow: hidden;
		aspect-ratio: 16 / 9;
		position: relative;
	}

	.output-frame {
		min-height: 360px;
	}

	video {
		width: 100%;
		height: 100%;
		display: block;
		object-fit: contain;
	}

	.hidden-video,
	.hidden-canvas {
		position: absolute;
		width: 1px;
		height: 1px;
		opacity: 0;
		pointer-events: none;
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

	.metrics {
		display: grid;
		grid-template-columns: repeat(3, minmax(0, 1fr));
		gap: 10px;
		margin: 14px 0;
	}

	.metrics > div,
	.live-metrics > div,
	.empty {
		border: 1px solid #26302c;
		background: #161b1a;
		border-radius: 8px;
	}

	.metrics > div,
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
		margin-top: 14px;
	}

	.detail-grid {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 14px;
	}

	.bars {
		display: grid;
		gap: 12px;
	}

	.bar-label {
		display: flex;
		justify-content: space-between;
		gap: 12px;
		margin-bottom: 6px;
		font-size: 14px;
	}

	.bar-track {
		height: 8px;
		background: #27312d;
		border-radius: 999px;
		overflow: hidden;
	}

	.bar-track div {
		height: 100%;
		background: #77c996;
	}

	code {
		display: block;
		white-space: pre-wrap;
		overflow-wrap: anywhere;
		padding: 12px;
		margin-bottom: 14px;
		background: #0d1110;
		border: 1px solid #26302c;
		border-radius: 6px;
		color: #d8e4dd;
		font-size: 13px;
	}

	.empty {
		padding: 24px;
		color: #c4ccc8;
	}

	.error {
		color: #ffc2b8;
	}

	.modal-backdrop {
		position: fixed;
		inset: 0;
		background: rgba(0, 0, 0, 0.75);
		display: grid;
		place-items: center;
		z-index: 100;
	}

	.modal {
		background: #161b1a;
		border: 1px solid #26302c;
		border-radius: 10px;
		padding: 24px;
		width: min(420px, 90vw);
	}

	.modal .status {
		color: #c4ccc8;
		margin: 8px 0 16px;
	}

	.progress-track {
		height: 10px;
		background: #27312d;
		border-radius: 999px;
		overflow: hidden;
		margin-bottom: 10px;
	}

	.progress-fill {
		height: 100%;
		background: #77c996;
		transition: width 0.2s ease;
	}

	@media (max-width: 900px) {
		.shell {
			padding: 16px;
		}

		.workspace,
		.detail-grid,
		.full-form .form-grid,
		.live-form .form-grid,
		.live-metrics {
			grid-template-columns: 1fr;
		}

		.field.span-2 {
			grid-column: auto;
		}

		.metrics {
			grid-template-columns: 1fr;
		}
	}
</style>
