<script lang="ts">
	let {
		canvas = $bindable<HTMLCanvasElement | null>(null),
		children,
	}: {
		canvas?: HTMLCanvasElement | null;
		children?: import('svelte').Snippet;
	} = $props();
</script>

<section class="live-stage">
	<div class="video-frame output-frame">
		<canvas bind:this={canvas} class="output-canvas"></canvas>
		{#if children}
			{@render children()}
		{/if}
	</div>
</section>

<style>
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

	@media (max-width: 980px) {
		.video-frame {
			aspect-ratio: 16 / 9;
			min-height: 280px;
		}
	}
</style>
