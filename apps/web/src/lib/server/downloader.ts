import { spawn } from 'node:child_process';
import { mkdir } from 'node:fs/promises';
import { dirname } from 'node:path';
import { workspaceRoot } from './paths';

export type DownloadProgress = {
	stage: 'download';
	percent: number;
	eta?: string;
	speed?: string;
	message: string;
};

function parseYtDlpProgress(line: string): Partial<DownloadProgress> | null {
	// [download]  12.3% of ~10.00MiB at  1.00MiB/s ETA 00:00:10
	const match = /\[download\]\s+([\d.]+)%\s+of\s+([^\s]+)\s+(?:at\s+([^\s]+)\s+)?(?:ETA\s+(\S+))?/i.exec(line);
	if (!match) return null;
	return {
		percent: Number.parseFloat(match[1]),
		message: line.trim(),
		eta: match[4],
		speed: match[3]
	};
}

export async function downloadYouTube(
	youtubeUrl: string,
	outputPath: string,
	{
		start,
		end,
		onProgress
	}: {
		start?: string;
		end?: string;
		onProgress?: (progress: DownloadProgress) => void;
	}
): Promise<void> {
	await mkdir(dirname(outputPath), { recursive: true });

	const downloadSections = start || end ? `*${start ?? '00:00:00'}-${end ?? '00:02:00'}` : '*00:00:00-00:02:00';

	const args = [
		'--no-playlist',
		'--download-sections',
		downloadSections,
		'-f',
		'bv*[vcodec^=avc1][height<=720]+ba[ext=m4a]/b[ext=mp4][height<=720]/b',
		'--merge-output-format',
		'mp4',
		'--recode-video',
		'mp4',
		'--postprocessor-args',
		'-c:v libx264 -crf 23 -c:a aac',
		'--newline',
		'--progress',
		'-o',
		outputPath,
		youtubeUrl
	];

	return new Promise((resolve, reject) => {
		const proc = spawn('yt-dlp', args, {
			cwd: workspaceRoot(),
			env: { ...process.env }
		});

		let stderr = '';

		proc.stdout?.on('data', (chunk: Buffer) => {
			const line = chunk.toString();
			const progress = parseYtDlpProgress(line);
			if (progress && onProgress) {
				onProgress({
					stage: 'download',
					percent: progress.percent ?? 0,
					eta: progress.eta,
					speed: progress.speed,
					message: progress.message ?? line.trim()
				});
			}
		});

		proc.stderr?.on('data', (chunk: Buffer) => {
			stderr += chunk.toString();
			const progress = parseYtDlpProgress(stderr);
			if (progress && onProgress) {
				onProgress({
					stage: 'download',
					percent: progress.percent ?? 0,
					eta: progress.eta,
					speed: progress.speed,
					message: progress.message ?? stderr.trim()
				});
			}
		});

		proc.on('error', (err) => {
			reject(new Error(`Failed to start yt-dlp: ${err.message}`));
		});

		proc.on('close', (code) => {
			if (code === 0) {
				resolve();
			} else {
				reject(new Error(`yt-dlp exited with code ${code}: ${stderr || 'unknown error'}`));
			}
		});
	});
}
