import { json } from '@sveltejs/kit';
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { basename, join } from 'node:path';

const outputDir = join(process.cwd(), 'data', 'outputs');

type RunSummary = {
	id: string;
	label: string;
	video: string;
	csv: string | null;
	sizeBytes: number;
	durationLabel: string;
	detections: number;
	classes: Record<string, number>;
};

function matchingCsv(videoName: string): string | null {
	const stem = videoName.replace(/\.mp4$/, '');
	const candidates = [
		`${stem}.csv`,
		`${stem.replace(/overlay/, 'positions')}.csv`,
		`${stem.replace(/player_overlay/, 'player_positions')}.csv`,
		`${stem.replace(/football_yolov5_overlay/, 'football_yolov5_positions')}.csv`
	];

	for (const candidate of candidates) {
		if (existsSync(join(outputDir, candidate))) return candidate;
	}

	return null;
}

function summarizeCsv(csvName: string | null): Pick<RunSummary, 'detections' | 'classes'> {
	if (!csvName) return { detections: 0, classes: {} };

	const path = join(outputDir, basename(csvName));
	if (!existsSync(path)) return { detections: 0, classes: {} };

	const lines = readFileSync(path, 'utf8').trim().split(/\r?\n/);
	if (lines.length <= 1) return { detections: 0, classes: {} };

	const header = lines[0].split(',');
	const classIndex = header.indexOf('class_name');
	const classes: Record<string, number> = {};

	for (const line of lines.slice(1)) {
		if (classIndex === -1) {
			classes.player = (classes.player ?? 0) + 1;
			continue;
		}
		const columns = line.split(',');
		const className = columns[classIndex] || 'unknown';
		classes[className] = (classes[className] ?? 0) + 1;
	}

	return { detections: lines.length - 1, classes };
}

function labelFor(videoName: string) {
	return videoName
		.replace(/\.mp4$/, '')
		.replaceAll('_', ' ')
		.replace(/\b\w/g, (char) => char.toUpperCase());
}

export function GET() {
	if (!existsSync(outputDir)) return json({ runs: [] });

	const runs = readdirSync(outputDir)
		.filter((file) => file.endsWith('.mp4'))
		.map((videoName): RunSummary => {
			const csv = matchingCsv(videoName);
			const summary = summarizeCsv(csv);
			const stats = statSync(join(outputDir, videoName));

			return {
				id: videoName,
				label: labelFor(videoName),
				video: `/media/${videoName}`,
				csv: csv ? `/media/${csv}` : null,
				sizeBytes: stats.size,
				durationLabel: 'generated overlay',
				...summary
			};
		})
		.sort((a, b) => b.sizeBytes - a.sizeBytes);

	return json({ runs });
}
