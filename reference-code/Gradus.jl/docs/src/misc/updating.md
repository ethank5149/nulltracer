# Updating

In general, it is sufficient to use Pkg to update Gradus.jl provided you
installed it using `add`:
```julia
pkg> up Gradus
```

## December 2025: migrating from GitHub to Codeberg

The Gradus.jl source code moved from GitHub to Codeberg in December 2025. It is
recommended if you installed Gradus before December to follow the migration
procedure to fetch updates from the correct location:

```julia
julia>]
# update the registries
pkg> registry up
pkg> up Gradus
```

If, for whatever reason, that should fail, consider deleting the local Gradus
package and re-adding it:
```julia
pkg> rm Gradus
```
and then in the terminal:
```
rm -rf ~/.julia/packages/Gradus ~/.julia/compiled/v1.12/Gradus
```

If you have a dev version of Gradus (`~/.julia/dev/Gradus`), you can manually
update the origin to point to Codeberg repository:

```bash
git remote set-url origin https://codeberg.org/astro-group/Gradus.jl
```
