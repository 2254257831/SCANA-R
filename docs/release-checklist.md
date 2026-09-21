# Transition from local preparation to a public release

No step in this document has published the repository. The local Git repository
has no remote. Its source and evidence bundle are prepared separately.

Before a later user-authorized upload:

1. Supply the actual GitHub username, repository name, authors/affiliations and
   public manuscript record. The registration email alone does not identify a
   GitHub username. Do not fabricate formal BibTeX metadata.
2. Review the local commit author email before pushing history. The project owner
   provided an email for this dedicated account; a verified GitHub
   noreply address can replace it if desired. No email is placed on the homepage.
3. Keep `artifacts/`, `outputs/`, caches and environments out of Git. Distribute the
   hash-matched large artifact zip as a separate release asset or data repository;
   then document the actual download URL in the README and homepage.
4. Review the Apache-2.0 LICENSE and third-party ACT MIT notice already included.
5. After explicit upload authorization, create/configure the chosen repository
   and use GitHub Pages from branch `main`, folder `/docs`. This project contains
   no automated deployment job that could publish unexpectedly.
6. Verify the official Pages URL and all resources. Replace the local homepage
   hyperlink at the end of the Chinese abstract with that verified HTTPS URL in
   a new manuscript copy; align the English manuscript and submission materials.

The paper's temporary file link is intentionally only useful on this computer.
It must not be used as the final published project URL.
