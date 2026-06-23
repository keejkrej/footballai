<script lang="ts">
	import { buildLiveSource, sourcePlaceholder } from '../live/source';
	import type { Device, LiveSource } from '../types';

	let {
		liveRunning,
		liveError,
		onStart,
		onStop,
		onError,
	}: {
		liveRunning: boolean;
		liveError: string;
		onStart: (payload: { source: LiveSource; device: Device }) => void;
		onStop: () => void;
		onError: (message: string) => void;
	} = $props();

	let device = $state<Device>('cuda');
	let liveSourceType = $state<'file' | 'youtube' | 'obs'>('file');
	let liveSourceValue = $state('');
	let liveFileStart = $state('00:00:00');
	let liveFileEnd = $state('00:02:00');
	let liveYoutubeStart = $state('00:00:00');
	let liveYoutubeEnd = $state('00:02:00');
	let liveMaxFps = $state(5);

	function handleStart() {
		if (!liveSourceValue.trim()) {
			onError('Please enter a source value');
			return;
		}

		const source = buildLiveSource({
			type: liveSourceType,
			value: liveSourceValue,
			fileStart: liveFileStart,
			fileEnd: liveFileEnd,
			youtubeStart: liveYoutubeStart,
			youtubeEnd: liveYoutubeEnd,
			maxFps: liveMaxFps,
		});

		if (source) {
			onStart({ source, device });
		}
	}
</script>

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
			<input type="text" bind:value={liveSourceValue} placeholder={sourcePlaceholder(liveSourceType)} />
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
			<button type="button" class="secondary" onclick={onStop}>Stop live</button>
		{:else}
			<button type="button" class="primary" onclick={handleStart}>Start live</button>
		{/if}
	</div>
	{#if liveError}
		<p class="error">{liveError}</p>
	{/if}
</section>

<style>
	h2 {
		font-family: var(--font-ui);
		font-size: 15px;
		font-weight: 500;
		line-height: 1.35;
		margin: 0 0 var(--space-3);
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

	.error {
		color: var(--error);
		margin-top: var(--space-3);
	}
</style>
