<script lang="ts">
	import { onMount } from 'svelte';
	import { Topbar, LiveForm, LiveMetrics, LiveStage } from './lib/components';
	import { useLiveStream } from './lib/live/stream.svelte';

	const stream = useLiveStream();
	let canvas = $state<HTMLCanvasElement | null>(null);

	$effect(() => {
		stream.bindCanvas(canvas);
	});

	onMount(() => {
		return () => stream.stop();
	});
</script>

<main class="shell">
	<aside class="left-pane">
		<Topbar />
		<LiveForm
			liveRunning={stream.liveRunning}
			liveError={stream.liveError}
			onStart={({ source, device }) => stream.start(source, device)}
			onStop={() => stream.stop()}
			onError={(message) => stream.setError(message)}
		/>
	</aside>

	<section class="right-pane">
		<LiveMetrics liveMeta={stream.liveMeta} />
		<LiveStage bind:canvas>
			{#if !stream.liveMeta}
				<div class="placeholder">
					<span>Annotated stream will appear here</span>
				</div>
			{/if}
		</LiveStage>
	</section>
</main>

<style>
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
	}
</style>
