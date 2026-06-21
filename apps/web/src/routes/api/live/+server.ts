import { json } from '@sveltejs/kit';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { workspaceRoot } from '$lib/server/paths';

const liveStatePath = join(workspaceRoot(), 'data', 'live', 'latest.json');

export function GET() {
	if (!existsSync(liveStatePath)) {
		return json({
			status: 'idle',
			message: 'No live inference snapshot found. Start uv run footballai-live.'
		});
	}

	return json(JSON.parse(readFileSync(liveStatePath, 'utf8')));
}
