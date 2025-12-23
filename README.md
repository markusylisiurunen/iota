# iota

a small typescript library.

## installation

```sh
npm install @markusylisiurunen/iota
```

## usage

```ts
import { iota } from "@markusylisiurunen/iota";

console.log(iota);
```

## development

iota requires Node.js 20+.

```sh
npm install
npm run check
npm run build
```

## creating a release

publishing to npm happens automatically via github actions when a github release is published.

release steps:

- run checks and build:

```sh
npm run check
npm run build
```

- bump the version (creates a git tag):

```sh
npm version patch
```

- push the commit and tag:

```sh
git push --follow-tags
```

- create a github release (this triggers the publish workflow):

```sh
gh release create v$(node -p "require('./package.json').version") --generate-notes
```
