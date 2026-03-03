package sk.ainet.embedded.coralnpu.codegen

/**
 * Helper for building C source code with proper indentation.
 */
class CCodeBuilder {
    private val sb = StringBuilder()
    private var indentLevel = 0
    private val indentStr = "  " // 2-space indent, matching Python codegen output

    /** Append a line at the current indentation level. */
    fun line(text: String): CCodeBuilder {
        sb.append(indentStr.repeat(indentLevel))
        sb.appendLine(text)
        return this
    }

    /** Append an empty line. */
    fun blankLine(): CCodeBuilder {
        sb.appendLine()
        return this
    }

    /** Append a line at zero indentation (top-level declarations). */
    fun topLevel(text: String): CCodeBuilder {
        sb.appendLine(text)
        return this
    }

    /** Increase indentation. */
    fun indent(): CCodeBuilder {
        indentLevel++
        return this
    }

    /** Decrease indentation. */
    fun dedent(): CCodeBuilder {
        require(indentLevel > 0) { "Cannot dedent below 0" }
        indentLevel--
        return this
    }

    /** Append raw text without indentation or newline. */
    fun raw(text: String): CCodeBuilder {
        sb.append(text)
        return this
    }

    override fun toString(): String = sb.toString()
}
