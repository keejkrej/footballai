<script lang="ts">
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

	let runs = $state<RunSummary[]>([]);
	let selectedId = $state('');
	let loading = $state(true);
	let error = $state('');

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

	$effect(() => {
		loadRuns();
	});
</script>

<svelte:head>
	<title>FootballAI Overlay Lab</title>
	<meta
		name="description"
		content="Review football player detection overlays, model outputs, and pipeline commands."
	/>
</svelte:head>

<main class="shell">
	<header class="topbar">
		<div>
			<p class="eyebrow">FootballAI</p>
			<h1>Overlay Lab</h1>
		</div>
		<button type="button" onclick={loadRuns}>Refresh</button>
	</header>

	{#if loading}
		<section class="empty">Loading generated overlays...</section>
	{:else if error}
		<section class="empty error">{error}</section>
	{:else if runs.length === 0}
		<section class="empty">No overlay videos found in data/outputs.</section>
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
							python scripts/football_yolov5_overlay.py --video data/raw/youtube_clip.mp4 --weights
							models/football_yolov5_best.pt --output data/outputs/football_yolov5_overlay.mp4
						</code>
						{#if selectedRun?.csv}
							<a href={selectedRun.csv}>Download CSV</a>
						{/if}
					</section>
				</div>
			</section>
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
	}

	video {
		width: 100%;
		height: 100%;
		display: block;
		object-fit: contain;
	}

	.metrics {
		display: grid;
		grid-template-columns: repeat(3, minmax(0, 1fr));
		gap: 10px;
		margin: 14px 0;
	}

	.metrics > div,
	.panel,
	.empty {
		border: 1px solid #26302c;
		background: #161b1a;
		border-radius: 8px;
	}

	.metrics > div {
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

	.detail-grid {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 14px;
	}

	.panel {
		padding: 16px;
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

	@media (max-width: 900px) {
		.shell {
			padding: 16px;
		}

		.workspace,
		.detail-grid {
			grid-template-columns: 1fr;
		}

		.metrics {
			grid-template-columns: 1fr;
		}
	}
</style>
