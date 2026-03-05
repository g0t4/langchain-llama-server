abbr build "uv build"
abbr bump_patch "uvx bump-my-version bump patch; git push && git push --tags"
function get_pyproject_current_version
    cat pyproject.toml | grep current_version | string match --regex "\d+\.\d+\.\d+"
end
abbr create_release 'gh release create v$(get_pyproject_current_version) --title "v$(get_pyproject_current_version)"'
abbr aio "uvx bump-my-version bump patch; git push && git push --tags && gh release create v\$(get_pyproject_current_version) --title \"v\$(get_pyproject_current_version)\" --generate-notes"
