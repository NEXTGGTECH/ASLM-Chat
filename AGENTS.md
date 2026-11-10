# AGENTS.md

Rules for AI agents working in this repository. C# has its own rules; every other language (Python,
JavaScript, ...) follows the rules for other languages.

## Work

- Never rush and never be lazy. Gather the information you need and fact-check it against the code.
- Follow the existing structure and the style of the code and comments.
- Quality comes first: no workarounds, hacks, or sloppy code.
- Write everything in the codebase in English unless instructed otherwise.
- Never create files the project does not need (README.md, LICENSE.md, documentation, notes, reports)
  without a direct instruction.
- Keep temporary files (bulk-edit scripts, intermediate output) outside the repository and delete them
  once their task is done.
- Never wipe build output or databases without the user's confirmation. Building over the previous
  output is fine; a clean rebuild that deletes the existing build data is not. Updating data and
  applying migrations is fine; dropping a database with its data is not.
- Builds and databases created only to verify changes, which did not exist before, are exempt: anything
  may be done with them. Build them into a custom output folder set through a build argument, never
  the default one, and delete them once the task no longer needs them.
- Split large tasks into stages; report after each stage and wait for confirmation before the next one.

## Code and comments

### All languages

- A source file starts with the license header as a line comment, then an empty line:
  `Copyright <owner>. <license>.`, for example `// Copyright Example. Apache License 2.0.`
  Copy the owner and license from the headers of existing files; without them, take them from the
  license file. Without a license file the license is `All Rights Reserved` and the owner is the owner
  of the repository.
- UTF-8 without BOM, no trailing whitespace, no tabs, one final newline. Keep each file's line endings.
- Every class or type and every function or method, private and nested ones included, has a comment
  directly above it, with no empty line between.
- Comments are full sentences ending with a period. They state the purpose and the non-obvious behavior
  (what and why); they never restate a name or narrate obvious lines.
- Functions of about 25+ lines are split into logical steps by block comments, functions of 60+ lines
  get a comment on every major step, shorter ones only where a step is not obvious.
- A block comment inside a body is preceded by an empty line unless it is the first line of a block.
  It uses the mood the surrounding code uses.
- No comments at the end of code lines (tool pragmas excepted), no decorative separators, no TODOs, no
  volatile facts ("currently unused", caller counts).
- An exception handler that ignores the error says why that is safe.
- Section headers name groups of members: a lone comment line in sentence case without a period
  (`// Fields`, `# Engine lookup`). Larger classes and modules use them; small ones need none.
- Comments stay true: update them together with the code. A test comment states exactly what the test
  checks.

### C#

- Every type and member has a multi-line `/// <summary>`, private fields, constants and enum members
  included. Third person ("Returns ..."), placed above the attributes. `<param>` and `<returns>` only
  where the surrounding code uses them. The XML stays well-formed (`&lt;`, `<c>...</c>`).
- A local function gets a `//` comment line above it.
- Test summaries keep the form the project already uses ("Verifies that ..." or a statement of the
  checked behavior).
- Block-scoped namespaces.
- Section headers: two empty lines before (none right after an opening brace), one after. One empty line
  between members.
- Comment lines up to 120 characters.

### Python, JavaScript and other languages

- No docstrings or doc-comment blocks (Python docstrings, JSDoc `/** */`): plain line comments only,
  placed above the decorators.
- Functions use the imperative mood ("Return ..."), tests "Verify that ...", classes a noun phrase saying
  what the class is.
- Variables, constants and fields get a comment when their meaning is not obvious; a group of related
  assignments shares one comment. Do not overdo it.
- Section headers: two empty lines before and one after at module level, one before and one after inside
  a class.
- Other empty lines follow the language standard (PEP 8 for Python): never two in a row inside a function
  or class, none right after a definition line.
- Comment lines up to 115 characters.
