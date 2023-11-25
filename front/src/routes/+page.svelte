<script lang="ts">
	import { MLModel } from '$lib/business_logic/models';
	import { Model, getModels, getResult, sendFrame } from '$lib/api';
	import Button from '$lib/components/Button.svelte';
	import DropdownSelector from '$lib/components/DropdownSelector.svelte';
	import '$lib/styles.css';
	import { onMount } from 'svelte';

	/**
	 * If this is true, capture feed from the local web camera and stream it to the ML model.
	 */
	let isStreaming: boolean = false;
	let isLoading: boolean = false;
	let videoSource: HTMLVideoElement;

	let canvasElement: HTMLCanvasElement;
	let ctx: CanvasRenderingContext2D;

	let currentAction: string | null = null;

	const startStreaming = async () => {
		isStreaming = true;

		try {
			isLoading = true;
			const stream = await navigator.mediaDevices.getUserMedia({
				video: true
			});
			videoSource.srcObject = stream;
			videoSource.play();
			isLoading = false;
			startCapture();
		} catch (error) {
			console.log(error);
		}
	};

	const stopStreaming = () => {
		isStreaming = false;

		try {
			videoSource.srcObject?.getTracks().forEach((track) => track.stop());
		} catch (error) {
			console.log(error);
		}
	};

	let models: Model[] | null = null;
	let currentModel: Model | null = null;

	function startCapture() {
		console.log('Start Video Capture');

		navigator.mediaDevices
			.getUserMedia({ video: true })
			.then((stream) => {
				videoSource.srcObject = stream;
				captureFrame();
			})
			.catch((error) => {
				console.error('Error accessing camera:', error);
			});
	}

	async function captureFrame() {
		console.log('Start Frame Capture');
		let ctx = canvasElement.getContext('2d');

		ctx.drawImage(videoSource, 0, 0, canvasElement.width, canvasElement.height);
		console.log('Drawing Image Frame Capture');
		const frameData = canvasElement.toDataURL('image/jpeg');
		let res = await sendFrame(frameData);
		if (res !== undefined && res.got_video === true && currentModel !== null) {
			let res2 = await getResult(res.id, currentModel.id);
			if (res2 !== undefined) {
				currentAction = res2.result;
			}
		}

		requestAnimationFrame(captureFrame);
	}

	onMount(async () => {
		models = await getModels();
		currentModel = models[0];
	});
</script>

<div class="container">
	<h1>PMLDL Visualization</h1>
	<h2>Select model</h2>
	{#if models === null}
		Loading models...
	{:else}
		<p>Our project implements several models. Please select the one you want to test now.</p>
		<DropdownSelector bind:values={models} bind:value={currentModel} />
	{/if}

	<h2>Upload video</h2>
	<p>You can upload a video to perform an offline action recognition on it.</p>
	<div class="upload-video-container">
		<input type="file" id="video" name="video" accept="video/*" />
		<Button buttonType="primary">Upload</Button>
	</div>

	<h2>Real-time streaming</h2>
	<p>
		You can use web camera of this computer to stream it to the ML model and analyze your current
		action! (I don't promise it won't be "staring at the screen")
	</p>
	{#if isStreaming}
		<Button buttonType="danger" on:click={stopStreaming}>Stop streaming</Button>
	{:else}
		<Button buttonType="primary" on:click={startStreaming}>Start streaming</Button>
	{/if}

	<h3>As sent to the model</h3>
	<canvas bind:this={canvasElement} id="canvas" width="320" height="240" />
	{#if currentAction !== null}
		<p class="action">{currentAction}</p>
	{/if}
	{#if isStreaming}
		<h3>Original video</h3>
		<video style="margin-top: 8px;" class="video-feed" bind:this={videoSource} autoplay loop muted
		></video>
	{/if}
</div>

<style>
	.container {
		max-width: 600px;
		min-height: 100svh;
		box-sizing: border-box;
		margin: 0 auto;
		padding: 1em;
	}

	.video-feed {
		width: 100%;
		height: auto;
		border-radius: 8px;
	}

	.upload-video-container {
		display: flex;
		flex-direction: column;
		gap: 8px;
		border: 2px solid var(--border-color);
		padding: 8px;
		border-radius: 8px;
	}
</style>
