import { json } from '@sveltejs/kit';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

const liveStatePath = join(process.cwd(), 'data', 'live', 'latest.json');

export function GET() {
	if (!existsSync(liveStatePath)) {
		return json({
			status: 'idle',
			message: 'No live inference snapshot found. Start scripts/live_stream_inference.py.'
		});
	}

	return json(JSON.parse(readFileSync(liveStatePath, 'utf8')));
}
