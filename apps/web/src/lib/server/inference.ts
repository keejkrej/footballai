import { spawn } from 'node:child_process';
import { workspaceRoot } from './paths';

export type InferenceProgress =
	| {
			stage: 'inference';
			frame: number;
			processed: number;
			total: number | null;
			classes: Record<string, number>;
	  }
	| {
			stage: 'done';
			output: string;
			csv: string;
			processed: number;
	  }
	| {
			stage: 'error';
			message: string;
	  };

export function runFullInference(
	inputPath: string,
	outputPath: string,
	csvPath: string,
	{
		onProgress,
		onError,
		onDone,
		device = 'cuda',
		maxFrames = 0,
		stride = 1,
		skipTeamFit = false
	}: {
		onProgress?: (progress: InferenceProgress) => void;
		onError?: (error: Error) => void;
		onDone?: (output: string, csv: string, processed: number) => void;
		device?: string;
		maxFrames?: number;
		stride?: number;
		skipTeamFit?: boolean;
	}
): { kill: () => void } {
	const args = [
		'run',
		'inference',
		'full',
		'--input',
		inputPath,
		'--output',
		outputPath,
		'--csv',
		csvPath,
		'--device',
		device,
		'--stride',
		String(stride),
		'--max-frames',
		String(maxFrames)
	];
	if (skipTeamFit) args.push('--skip-team-fit');

	const proc = spawn('uv', args, {
		cwd: workspaceRoot(),
		env: { ...process.env }
	});

	let buffer = '';

	const handleLine = (line: string) => {
		line = line.trim();
		if (!line) return;

		// Try to parse JSON progress lines emitted by the Python pipeline.
		try {
			if (line.startsWith('{')) {
				const payload = JSON.parse(line) as InferenceProgress;
				if (payload.stage === 'done' && onDone) {
					onDone(payload.output, payload.csv, payload.processed);
				} else if (onProgress) {
					onProgress(payload);
				}
				return;
			}
		} catch {
			// not JSON; fall through to text fallback
		}

		if (onProgress) {
			onProgress({ stage: 'inference', frame: 0, processed: 0, total: null, classes: {} });
		}
	};

	const appendChunk = (chunk: Buffer) => {
		buffer += chunk.toString();
		let idx: number;
		while ((idx = buffer.indexOf('\n')) !== -1) {
			const line = buffer.slice(0, idx);
			buffer = buffer.slice(idx + 1);
			handleLine(line);
		}
	};

	proc.stdout?.on('data', appendChunk);
	proc.stderr?.on('data', appendChunk);

	proc.on('error', (err) => {
		if (onError) onError(new Error(`Failed to start inference: ${err.message}`));
	});

	proc.on('close', (code) => {
		if (code !== 0) {
			if (onError) onError(new Error(`Inference process exited with code ${code}`));
		}
	});

	return {
		kill: () => {
			if (!proc.killed) proc.kill('SIGTERM');
		}
	};
}
