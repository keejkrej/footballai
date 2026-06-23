<script lang="ts">
	import ThemeToggle from './ThemeToggle.svelte';
</script>

<header class="topbar">
	<h1>FootballAI</h1>
	<div class="topbar-tools">
		<ThemeToggle />
	</div>
</header>

<style>
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
		margin: 0;
	}

	.topbar-tools {
		display: flex;
		align-items: center;
		gap: var(--space-2);
	}
</style>
