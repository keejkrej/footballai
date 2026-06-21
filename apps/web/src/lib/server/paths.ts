import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * Resolve the monorepo root from any file inside apps/web.
 *
 * The helper walks upward until it finds pyproject.toml containing the uv
 * workspace marker, then falls back to the first directory containing a .git
 * folder.
 */
export function workspaceRoot(): string {
	let current = dirname(fileURLToPath(import.meta.url));

	while (current !== dirname(current)) {
		const pyproject = resolve(current, 'pyproject.toml');
		if (existsSync(pyproject)) {
			try {
				const contents = readFileSync(pyproject, 'utf8');
				if (contents.includes('[tool.uv.workspace]')) {
					return current;
				}
			} catch {
				// ignore and keep walking
			}
		}
		if (existsSync(resolve(current, '.git'))) {
			return current;
		}
		current = dirname(current);
	}

	// Final fallback: four levels up from apps/web/src/lib/server/paths.ts
	return resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..', '..');
}
