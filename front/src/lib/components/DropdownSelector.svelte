<script lang="ts">
	import DropdownSelectorArrow from '$lib/icons/DropdownSelectorArrow.svelte';
	import { fly } from 'svelte/transition';

	export let is_open = false;
	export let values: any[] = [];
	export let value: any | null = null;

	export let selectorButton: any | undefined = undefined;

	function outClick(event) {
		if (selectorButton !== undefined && event.target !== selectorButton) {
			is_open = false;
		}
	}
</script>

<svelte:window on:click={outClick} />

<div style="user-select: none">
	<div bind:this={selectorButton} class="selector" on:click={() => (is_open = !is_open)}>
		<span style="pointer-events: none;">
			{#if value}
				{value}
			{:else}
				Select...
			{/if}
		</span>
		<DropdownSelectorArrow
			style="
			pointer-events: none;
			width: 14px;
			height: 14px;
			right: 2px;
			margin: -2px -6px -2px 4px;
			padding: 3px;
			background-color: var(--color-theme-1);
			border-radius: 4px;
			color: var(--color-text);
		"
		/>
	</div>

	{#if is_open}
		<div class="list" in:fly={{ y: -5, duration: 200 }}>
			{#each values as val}
				<div
					class="list-item"
					on:click={() => {
						value = val;
						is_open = false;
					}}
				>
					{val}
				</div>
			{/each}
		</div>
	{/if}
</div>

<style>
	.selector {
		background-color: var(--color-bg-3);
		border-radius: 4px;
		padding: 4px 8px;
		cursor: pointer;
		display: flex;
		align-items: center;
		justify-content: space-between;
	}

	.list {
		position: absolute;
		background-color: var(--color-bg-3);
		border: 1px solid var(--border-color);
		border-radius: 4px;
		padding: 2px 2px;
		margin-top: 4px;
		width: fit-content;
		max-height: 200px;
		overflow-y: auto;
	}

	.list-item {
		cursor: pointer;
		padding: 4px 4px;
		border-radius: 3px;
		overflow-wrap: anywhere;
	}

	.list-item:hover {
		background-color: var(--color-bg-3-hover);
	}
</style>
