<script lang="ts">
	let theme = $state<'light' | 'dark'>(
		typeof window !== 'undefined' && window.matchMedia('(prefers-color-scheme: dark)').matches
			? 'dark'
			: 'light'
	);

	$effect(() => {
		document.documentElement.setAttribute('data-theme', theme);
	});

	function toggle() {
		theme = theme === 'light' ? 'dark' : 'light';
	}
</script>

<button
	type="button"
	class="theme-toggle"
	aria-label={theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}
	onclick={toggle}
>
	<span class="theme-icon" aria-hidden="true">{theme === 'light' ? '☀️' : '🌙'}</span>
	<span class="theme-label">{theme === 'light' ? 'Light' : 'Dark'}</span>
</button>

<style>
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
</style>
