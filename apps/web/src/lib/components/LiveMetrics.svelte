<script lang="ts">
	import type { LiveMetadata } from '../types';

	let { liveMeta }: { liveMeta: LiveMetadata | null } = $props();
</script>

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

<style>
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

	@media (max-width: 980px) {
		.live-metrics {
			grid-template-columns: repeat(2, minmax(0, 1fr));
		}
	}

	@media (max-width: 560px) {
		.live-metrics {
			grid-template-columns: 1fr;
		}
	}
</style>
