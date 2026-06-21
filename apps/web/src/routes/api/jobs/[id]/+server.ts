import { json } from '@sveltejs/kit';
import { getJob } from '$lib/server/jobs';

export async function GET({ params }) {
	const job = await getJob(params.id);
	if (!job) {
		return json({ error: 'Job not found' }, { status: 404 });
	}
	return json({ job });
}
