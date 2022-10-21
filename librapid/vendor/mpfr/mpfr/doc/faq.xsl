<?xml version="1.0"?>

<!DOCTYPE stylesheet [
<!ENTITY styles SYSTEM "visual.css">
<!ENTITY filter "h:div[@class = 'logo' or @class = 'end']">
]>

<!--
XSLT stylesheet to generate the FAQ.html file distributed in MPFR from
the faq.html file on the MPFR web site. See the update-faq script.
-->

<xsl:stylesheet version="1.0"
                xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:h="http://www.w3.org/1999/xhtml"
                xmlns="http://www.w3.org/1999/xhtml">

<xsl:output method="xml"
            encoding="iso-8859-1"
            doctype-public="-//W3C//DTD XHTML 1.0 Strict//EN"
            doctype-system="http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd"/>

<xsl:template match="/">
  <xsl:text>&#10;</xsl:text>
  <xsl:comment>
Copyright 2000-2022 Free Software Foundation, Inc.
Contributed by the AriC and Caramba projects, INRIA.

This file is part of the GNU MPFR Library.

The GNU MPFR Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The GNU MPFR Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the GNU MPFR Library; see the file COPYING.LESSER.  If not, see
https://www.gnu.org/licenses/ or write to the Free Software Foundation, Inc.,
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA.
</xsl:comment>
  <xsl:text>&#10;</xsl:text>
  <xsl:copy>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="/comment()"/>

<xsl:template match="node()|@*">
  <xsl:copy>
    <xsl:apply-templates select="node()|@*"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="h:head">
  <xsl:copy>
    <xsl:text>&#10;</xsl:text>
    <xsl:copy-of select="h:title"/>
    <xsl:text>&#10;</xsl:text>
    <style type="text/css"><xsl:text disable-output-escaping="yes">/*&lt;![CDATA[*/&#10;&styles;/*]]&gt;*/</xsl:text></style>
    <xsl:text>&#10;</xsl:text>
  </xsl:copy>
</xsl:template>

<!-- Note: the MPFR logo section is filtered out; this is important
     as the logo is non-free. -->
<xsl:template match="&filter; |
                     text()[preceding-sibling::*[1]/self::&filter;]"/>

</xsl:stylesheet>
