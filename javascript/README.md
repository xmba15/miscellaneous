## Installation:

- [Node Version Manager](https://github.com/nvm-sh/nvm):

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash

# Install a node version with
nvm install 18
nvm use 18

# List all available node versions with
nvm list
```

- ESLint:

```bash
npm install eslint --save-dev

# Create ESLint configuration file with
npx eslint --init
# ESLint configuration will be saved into eslint.config.js, or eslint.config.mjs, or .eslintrc.json
```

- Prettier: Formatter

```bash
npm install --save-dev prettier eslint-config-prettier eslint-plugin-prettier
```

- Sample configuration:

```bash
npm run lint
npm run format
```

  + eslint.config.mjs
```mjs
import globals from "globals";
import pluginJs from "@eslint/js";
import prettierPlugin from "eslint-plugin-prettier";
import prettierConfig from "eslint-config-prettier";

/** @type {import('eslint').Linter.Config[]} */
export default [
  {
    files: ["**/*.js"],
    languageOptions: {
      sourceType: "commonjs"
    }
  },
  {
    languageOptions: {
      globals: globals.browser
    }
  },
  pluginJs.configs.recommended, // Enable recommended JS rules
  prettierConfig, // Disable conflicting ESLint rules with Prettier
  {
    plugins: {
      prettier: prettierPlugin
    },
    rules: {
      "prettier/prettier": [
        "error",
        {
          singleQuote: true,
          semi: false,
          trailingComma: "es5"
        }
      ]
    }
  }
];
```

  + .prettierrc.json
```json
{
  "singleQuote": true,
  "semi": false,
  "trailingComma": "es5"
}
```

  + package.json

```json
"scripts": {
  "lint": "eslint .",
  "format": "prettier --write ."
}
```
