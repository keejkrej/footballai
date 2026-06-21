import { error } from '@sveltejs/kit';
import { createReadStream, existsSync, statSync } from 'node:fs';
import { basename, extname, join } from 'node:path';
import { Readable } from 'node:stream';
import { workspaceRoot } from '$lib/server/paths';

const outputDir = join(workspaceRoot(), 'data', 'outputs');
const contentTypes: Record<string, string> = {
	'.mp4': 'video/mp4',
	'.csv': 'text/csv; charset=utf-8'
};

export function GET({ params, request }) {
	const file = basename(params.file);
	const path = join(outputDir, file);
	const extension = extname(file);

	if (!existsSync(path) || !(extension in contentTypes)) {
		error(404, 'Media file not found');
	}

	const stat = statSync(path);
	const range = request.headers.get('range');

	if (extension === '.mp4' && range) {
		const match = /bytes=(\d*)-(\d*)/.exec(range);
		if (!match) error(416, 'Invalid range');

		const start = match[1] ? Number(match[1]) : 0;
		const end = match[2] ? Number(match[2]) : stat.size - 1;
		if (start >= stat.size || end >= stat.size || start > end) error(416, 'Range not satisfiable');

		const stream = Readable.toWeb(createReadStream(path, { start, end })) as ReadableStream;
		return new Response(stream, {
			status: 206,
			headers: {
				'content-type': contentTypes[extension],
				'content-length': String(end - start + 1),
				'content-range': `bytes ${start}-${end}/${stat.size}`,
				'accept-ranges': 'bytes',
				'cache-control': 'no-store'
			}
		});
	}

	const stream = Readable.toWeb(createReadStream(path)) as ReadableStream;
	return new Response(stream, {
		headers: {
			'content-type': contentTypes[extension],
			'content-length': String(stat.size),
			'accept-ranges': extension === '.mp4' ? 'bytes' : 'none',
			'cache-control': 'no-store'
		}
	});
}
