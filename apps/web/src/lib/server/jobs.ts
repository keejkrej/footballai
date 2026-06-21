import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { workspaceRoot } from './paths';

export type JobStatus =
	| 'pending'
	| 'downloading'
	| 'inferencing'
	| 'done'
	| 'error';

export type Job = {
	id: string;
	status: JobStatus;
	youtubeUrl: string;
	rawPath: string;
	outputPath: string;
	csvPath: string;
	videoUrl: string;
	csvUrl: string | null;
	progress: number;
	message: string;
	createdAt: number;
	updatedAt: number;
};

function jobsDir(): string {
	return join(workspaceRoot(), 'data', 'jobs');
}

export async function getJob(id: string): Promise<Job | null> {
	try {
		const text = await readFile(join(jobsDir(), `${id}.json`), 'utf8');
		return JSON.parse(text) as Job;
	} catch {
		return null;
	}
}

export async function saveJob(job: Job): Promise<void> {
	await mkdir(jobsDir(), { recursive: true });
	await writeFile(join(jobsDir(), `${job.id}.json`), JSON.stringify(job, null, 2));
}

export function makeJobId(): string {
	return `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}
