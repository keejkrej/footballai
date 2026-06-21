import { spawn } from 'node:child_process';
import { workspaceRoot } from './paths';

export type DownloadProgress = {
	stage: 'download';
	percent: number;
	eta?: string;
	speed?: string;
	message: string;
};

function parseYtDlpProgress(line: string): Partial<DownloadProgress> | null {
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
	const args = [
		'run',
		'inference',
		'download',
		youtubeUrl,
		'--start',
		start ?? '00:00:00',
		'--end',
		end ?? '00:02:00',
		'--output',
		outputPath
	];

	return new Promise((resolve, reject) => {
		const proc = spawn('uv', args, {
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
		});

		proc.on('error', (err) => {
			reject(new Error(`Failed to start download: ${err.message}`));
		});

		proc.on('close', (code) => {
			if (code === 0) {
				resolve();
			} else {
				reject(new Error(`Download exited with code ${code}: ${stderr || 'unknown error'}`));
			}
		});
	});
}
