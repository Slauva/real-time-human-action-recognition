<script lang="ts">
	type ButtonType = undefined | 'danger' | 'primary';

	let typeToColorMapping = new Map<ButtonType, string>([
		[undefined, 'var(--color-text)'],
		['danger', 'var(--color-danger-fg)'],
		['primary', 'var(--color-theme-1-fg)']
	]);

	let typeToBgColorMapping = new Map<ButtonType, string>([
		[undefined, 'var(--color-bg-3)'],
		['danger', 'var(--color-bg-3)'],
		['primary', 'var(--color-theme-1)']
	]);

	let typeToHoverBgColorMapping = new Map<ButtonType, string>([
		[undefined, 'var(--color-bg-3-hover)'],
		['danger', 'var(--color-bg-3-hover)'],
		['primary', 'var(--color-theme-1-hover)']
	]);

	export let buttonType: ButtonType = undefined;
	export let style = '';
	export let text: string | undefined = undefined;

	$: color = typeToColorMapping.get(buttonType);
	$: bgColor = typeToBgColorMapping.get(buttonType);
	$: hoverBgColor = typeToHoverBgColorMapping.get(buttonType);
</script>

<button
	class="btn btn-primary"
	style="--bg-color: {bgColor}; --color: {color}; --hover-bg-color: {hoverBgColor}; {style}"
	{...$$restProps}
	on:click
>
	<slot>
		{text}
	</slot>
</button>

<style>
	button {
		background-color: var(--bg-color);
		border: none;
		padding: 6px 8px;
		color: var(--color);
		border-radius: 4px;
		outline: none;
		outline-color: transparent;
		transition: background-color 0.1s linear;
		transition: outline-color 0.2s;
		cursor: pointer;
	}

	button:hover {
		background-color: var(--hover-bg-color);
	}

	button:focus {
		outline-color: var(--color-theme-1);
		outline-width: 2px;
		outline-style: solid;
	}
</style>
