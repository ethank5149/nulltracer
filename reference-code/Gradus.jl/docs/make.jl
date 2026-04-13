push!(LOAD_PATH, "src")

using Documenter
using Gradus



makedocs(
    modules = [Gradus],
    clean = true,
    sitename = "Gradus.jl Documentation",
    warnonly = true,
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting-started.md",
        "Examples" => "examples.md",
        "Reference & walkthroughs" => [
            "Catalogue of metrics" => "metrics.md",
            "Accretion geometry" => "accretion-geometry.md",
            "Energyshift" => "redshift.md",
            "Emissivity profiles" => "emissivity.md",
            "Line profiles" => "lineprofiles.md",
            "Point functions" => "point-functions.md",
            "Problems and solvers" => "problems-and-solvers.md",
        ],
        "Geodesics and integration" => [
            "Geodesic integration" => "geodesic-integration.md",
            "Adaptive tracing" => "adaptive-tracing.md",
            # "Parallelism and ensembles" => "parallelism.md",
            "Implementing new metrics" => "custom-metrics.md",
            "Special radii" => "special-radii.md",
        ],
        "API" => ["Gradus" => "api-documentation/Gradus.md"] |> sort,
        "Miscellaneous" => ["Updating Gradus" => "misc/updating.md"],
    ],
    repo = Remotes.GitLab("codeberg.org", "astro-group", "Gradus.jl"),
)

deploydocs(repo = "codeberg.org/astro-group/Gradus.jl.git", branch = "pages")
