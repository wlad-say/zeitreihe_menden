# Guidelines

##  Ziel

Dieses Repository folgt klaren Standards, um Codequalität, Lesbarkeit und Zusammenarbeit sicherzustellen.

---

## Branching-Strategie

* `main` → stabiler Code (immer deploybar)
* `dev` → Integrations-Branch
* Feature-Branches:

  * `feature/<name>`
  * `bugfix/<name>`
  * `hotfix/<name>`

---

## Commit Messages

Wir verwenden klare, strukturierte Commit-Nachrichten:

Format:
type(scope): kurze Beschreibung

Beispiele:

* feat(auth): add login functionality
* fix(api): handle null response
* docs(readme): update setup instructions

Typen:

* feat → neues Feature
* fix → Bugfix
* docs → Dokumentation
* refactor → Code-Verbesserung ohne Feature
* test → Tests

---

## Pull Requests

* Jeder PR muss:

  * verständlich beschrieben sein
  * einen klaren Zweck haben
  * klein und fokussiert sein
* Mindestens 1 Review erforderlich
* Keine direkten Commits auf `main`

---

## Tests

* Neue Features müssen getestet werden
* Bestehende Tests dürfen nicht brechen

---

## Code Style

* Einheitlicher Stil (z. B. Prettier / Black / ESLint)
* Keine unnötigen Kommentare
* Verständliche Variablennamen

---

## Sicherheit

* Keine Secrets im Code (`.env` verwenden)
* `.env` niemals committen
* Sensible Daten immer anonymisieren

---

## Dependencies

* Nur notwendige Libraries hinzufügen
* Regelmäßig Updates prüfen

---

## Allgemeine Regeln

* "Works on my machine" ist kein akzeptabler Zustand
* Code muss für andere verständlich sein
* Lieber einfache Lösungen als unnötig komplexe

---
