import { json } from '@sveltejs/kit';
import { basename, join } from 'node:path';
import { downloadYouTube } from '$lib/server/downloader';
import { runFullInference } from '$lib/server/inference';
import { getJob, makeJobId, saveJob, type Job } from '$lib/server/jobs';
import { workspaceRoot } from '$lib/server/paths';

export async function POST({ request }) {
	const body = (await request.json()) as {
		youtubeUrl: string;
		start?: string;
		end?: string;
		maxFrames?: number;
		stride?: number;
		device?: string;
		skipTeamFit?: boolean;
	};

	const youtubeUrl = body.youtubeUrl?.trim();
	if (!youtubeUrl) {
		return json({ error: 'YouTube URL is required' }, { status: 400 });
	}

	const id = makeJobId();
	const root = workspaceRoot();
	const rawPath = join(root, 'data', 'raw', `${id}.mp4`);
	const outputPath = join(root, 'data', 'outputs', `${id}_overlay.mp4`);
	const csvPath = join(root, 'data', 'outputs', `${id}.csv`);

	const job: Job = {
		id,
		status: 'pending',
		youtubeUrl,
		rawPath,
		outputPath,
		csvPath,
		videoUrl: `/media/${basename(outputPath)}`,
		csvUrl: `/media/${basename(csvPath)}`,
		progress: 0,
		message: 'Queued',
		createdAt: Date.now(),
		updatedAt: Date.now()
	};
	await saveJob(job);

	// Run the pipeline asynchronously; respond immediately with the job id.
	(async () => {
		try {
			job.status = 'downloading';
			job.message = 'Downloading from YouTube...';
			job.updatedAt = Date.now();
			await saveJob(job);

			await downloadYouTube(youtubeUrl, rawPath, {
				start: body.start,
				end: body.end,
				onProgress: (progress) => {
					job.progress = Math.round(progress.percent);
					job.message = progress.message;
					job.updatedAt = Date.now();
					void saveJob(job);
				}
			});

			job.status = 'inferencing';
			job.progress = 0;
			job.message = 'Running inference...';
			job.updatedAt = Date.now();
			await saveJob(job);

			runFullInference(rawPath, outputPath, csvPath, {
				device: body.device ?? 'cuda',
				maxFrames: body.maxFrames ?? 0,
				stride: body.stride ?? 1,
				skipTeamFit: body.skipTeamFit ?? false,
				onProgress: (progress) => {
					if (progress.stage === 'inference') {
						job.progress = progress.total
							? Math.round((progress.processed / progress.total) * 100)
							: progress.processed;
						job.message = `Inference: ${progress.processed} frames processed`;
					}
					job.updatedAt = Date.now();
					void saveJob(job);
				},
				onDone: (output, csv, processed) => {
					job.status = 'done';
					job.progress = 100;
					job.message = `Done: ${processed} frames processed`;
					job.outputPath = output;
					job.csvPath = csv;
					job.videoUrl = `/media/${basename(output)}`;
					job.csvUrl = `/media/${basename(csv)}`;
					job.updatedAt = Date.now();
					void saveJob(job);
				},
				onError: (err) => {
					job.status = 'error';
					job.message = err.message;
					job.updatedAt = Date.now();
					void saveJob(job);
				}
			});
		} catch (err) {
			const message = err instanceof Error ? err.message : String(err);
			const current = await getJob(id);
			if (current) {
				current.status = 'error';
				current.message = message;
				current.updatedAt = Date.now();
				await saveJob(current);
			}
		}
	})();

	return json({ jobId: id });
}
