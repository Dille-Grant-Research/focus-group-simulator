import { jsPDF } from "jspdf";
import {
  Document,
  HeadingLevel,
  Packer,
  Paragraph,
  Table,
  TableCell,
  TableRow,
  TextRun,
  WidthType,
} from "docx";
import type { SessionResultData } from "./api";

// ── Shared helpers ────────────────────────────────────────────────────────────

function safeFilename(sessionName: string | null | undefined, ext: string): string {
  const base = sessionName?.trim().replace(/[^a-z0-9_\-\s]/gi, "").trim() || "transcript";
  return `${base}.${ext}`;
}

function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// ── Stance CSV ────────────────────────────────────────────────────────────────

interface StanceEntry {
  position?: string;
  key_points?: string[];
  sentiment_score?: number;
  contribution_count?: number;
}

export function downloadStanceCSV(
  data: Record<string, unknown>,
  sessionName: string | null | undefined
): void {
  const rows: string[][] = [
    ["Question", "Participant", "Position", "Sentiment Score", "Turns", "Key Points"],
  ];

  for (const [question, participants] of Object.entries(data)) {
    const participantMap = participants as Record<string, StanceEntry>;
    for (const [name, stance] of Object.entries(participantMap ?? {})) {
      rows.push([
        question,
        name,
        stance.position ?? "",
        stance.sentiment_score != null ? stance.sentiment_score.toFixed(2) : "",
        stance.contribution_count != null ? String(stance.contribution_count) : "",
        (stance.key_points ?? []).join("; "),
      ]);
    }
  }

  const csv = rows.map((row) => row.map(csvCell).join(",")).join("\n");
  triggerDownload(new Blob([csv], { type: "text/csv" }), safeFilename(sessionName, "csv"));
}

function csvCell(value: string): string {
  if (/[",\n]/.test(value)) return `"${value.replace(/"/g, '""')}"`;
  return value;
}

// ── PDF (jsPDF direct download) ───────────────────────────────────────────────

export function downloadPDF(
  result: SessionResultData,
  includeNonVerbals: boolean,
  sessionName: string | null | undefined
): void {
  const doc = new jsPDF({ unit: "mm", format: "a4" });
  const PW = doc.internal.pageSize.getWidth();   // 210mm
  const PH = doc.internal.pageSize.getHeight();  // 297mm
  const ML = 20, MR = 20, MT = 25, MB = 22;
  const CW = PW - ML - MR;                       // 170mm content width

  let y = MT;
  let pageNum = 1;

  // line height in mm for a given pt size (1pt = 0.353mm, ×1.4 leading)
  function lh(pt: number): number { return pt * 0.494; }

  function addPageIfNeeded(needed: number): void {
    if (y + needed > PH - MB) {
      doc.setFontSize(8);
      doc.setFont("helvetica", "normal");
      doc.setTextColor(160, 160, 160);
      doc.text(String(pageNum), PW / 2, PH - 10, { align: "center" });
      doc.addPage();
      y = MT;
      pageNum++;
    }
  }

  function addText(
    content: string,
    opts: {
      pt: number;
      bold?: boolean;
      italic?: boolean;
      rgb?: [number, number, number];
      indent?: number;
      gap?: number;
    }
  ): void {
    const { pt, bold = false, italic = false, rgb = [17, 17, 17], indent = 0, gap = 2 } = opts;
    const style = bold && italic ? "bolditalic" : bold ? "bold" : italic ? "italic" : "normal";
    doc.setFontSize(pt);
    doc.setFont("helvetica", style);
    doc.setTextColor(rgb[0], rgb[1], rgb[2]);

    const lines = doc.splitTextToSize(content, CW - indent) as string[];
    const lineHeight = lh(pt);

    for (const line of lines) {
      addPageIfNeeded(lineHeight);
      doc.text(line, ML + indent, y);
      y += lineHeight;
    }
    y += gap;
  }

  function rule(): void {
    addPageIfNeeded(3);
    doc.setDrawColor(220, 220, 220);
    doc.line(ML, y, PW - MR, y);
    y += 4;
  }

  // ── Title ──────────────────────────────────────────────────────────────────
  const title = result.session_name || "Focus Group Transcript";
  addText(title, { pt: 18, bold: true, gap: 3 });

  const metaParts: string[] = [];
  if (result.completed_at) metaParts.push(`Completed: ${new Date(result.completed_at).toLocaleString()}`);
  metaParts.push(`${result.participants.length} participants`, `${result.total_responses} turns`);
  addText(metaParts.join("   |   "), { pt: 9, rgb: [110, 110, 110], gap: 4 });

  if (result.purpose) {
    addText(result.purpose, { pt: 10, italic: true, rgb: [80, 80, 80], indent: 5, gap: 5 });
  }

  rule();

  // ── Participants ────────────────────────────────────────────────────────────
  addText("PARTICIPANTS", { pt: 8, bold: true, rgb: [130, 130, 130], gap: 2 });
  for (const p of result.participants) {
    addText(`• ${p.name as string}`, { pt: 10, indent: 3, gap: 1 });
  }
  y += 3;
  rule();

  // ── Transcript ──────────────────────────────────────────────────────────────
  addText("TRANSCRIPT", { pt: 8, bold: true, rgb: [130, 130, 130], gap: 3 });

  const allQuestions = [
    ...result.question_results,
    ...(result.closing_result ? [result.closing_result] : []),
  ];

  for (let qi = 0; qi < allQuestions.length; qi++) {
    const qr = allQuestions[qi];
    const isClosing = result.closing_result != null && qi === allQuestions.length - 1;
    const label = isClosing ? "Closing" : `Q${qi + 1}`;

    y += 2;
    addText(label, { pt: 8, bold: true, rgb: [170, 0, 0], gap: 0.5 });
    addText(qr.question, { pt: 11, bold: true, gap: 3 });

    for (const r of qr.responses) {
      addText(`${r.speaker_name}  (turn ${r.turn_number})`, { pt: 10, bold: true, indent: 6, gap: 0.5 });
      addText(r.content, { pt: 10, indent: 6, gap: 1.5 });

      if (includeNonVerbals) {
        const reactions = qr.nonverbal_reactions.filter((nv) => nv.response_turn === r.turn_number);
        if (reactions.length > 0) {
          const txt = reactions.map((nv) => `[${nv.participant_name}: ${nv.reaction}]`).join("  ");
          addText(txt, { pt: 9, italic: true, rgb: [110, 110, 110], indent: 6, gap: 1.5 });
        }
      }
    }
    y += 2;
  }

  // ── Stance Analysis ────────────────────────────────────────────────────────
  if (result.stance_matrix_dump) {
    rule();
    addText("STANCE ANALYSIS", { pt: 8, bold: true, rgb: [130, 130, 130], gap: 3 });

    for (const [question, participants] of Object.entries(result.stance_matrix_dump)) {
      addText(question, { pt: 11, bold: true, gap: 2 });
      const pMap = participants as Record<string, StanceEntry>;

      for (const [name, s] of Object.entries(pMap ?? {})) {
        const pos = s.position ?? "—";
        const sent = s.sentiment_score != null ? s.sentiment_score.toFixed(1) : "—";
        const turns = String(s.contribution_count ?? 0);
        addText(`${name}`, { pt: 9, bold: true, indent: 4, gap: 0.5 });
        addText(`${pos}  ·  sentiment ${sent}  ·  ${turns} turns`, { pt: 9, rgb: [80, 80, 80], indent: 10, gap: 0.5 });
        if ((s.key_points ?? []).length > 0) {
          addText((s.key_points ?? []).join("; "), { pt: 9, italic: true, rgb: [100, 100, 100], indent: 10, gap: 1.5 });
        } else {
          y += 1.5;
        }
      }
      y += 3;
    }
  }

  // Page number on last page
  doc.setFontSize(8);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(160, 160, 160);
  doc.text(String(pageNum), PW / 2, PH - 10, { align: "center" });

  doc.save(safeFilename(sessionName, "pdf"));
}

// ── DOCX ──────────────────────────────────────────────────────────────────────

export async function downloadDocx(
  result: SessionResultData,
  includeNonVerbals: boolean,
  sessionName: string | null | undefined
): Promise<void> {
  const children: (Paragraph | Table)[] = [];
  const title = result.session_name || "Focus Group Transcript";

  // Title
  children.push(
    new Paragraph({
      text: title,
      heading: HeadingLevel.TITLE,
    })
  );

  if (result.purpose) {
    children.push(
      new Paragraph({
        children: [new TextRun({ text: result.purpose, italics: true })],
        spacing: { before: 120, after: 240 },
      })
    );
  }

  // Metadata line
  if (result.completed_at) {
    const meta = [
      `Completed: ${new Date(result.completed_at).toLocaleString()}`,
      `${result.participants.length} participants`,
      `${result.total_responses} turns`,
    ].join("   |   ");
    children.push(
      new Paragraph({
        children: [new TextRun({ text: meta, size: 18, color: "666666" })],
        spacing: { after: 240 },
      })
    );
  }

  // Participants
  children.push(
    new Paragraph({
      text: "Participants",
      heading: HeadingLevel.HEADING_1,
      spacing: { before: 240, after: 120 },
    })
  );
  for (const p of result.participants) {
    children.push(
      new Paragraph({
        text: `• ${p.name}`,
        spacing: { before: 40 },
      })
    );
  }

  // Transcript
  children.push(
    new Paragraph({
      text: "Transcript",
      heading: HeadingLevel.HEADING_1,
      spacing: { before: 360, after: 120 },
    })
  );

  const allQuestions = [
    ...result.question_results,
    ...(result.closing_result ? [result.closing_result] : []),
  ];

  allQuestions.forEach((qr, qi) => {
    const isClosing = result.closing_result != null && qi === allQuestions.length - 1;
    const label = isClosing ? "Closing" : `Q${qi + 1}`;

    // Question heading
    children.push(
      new Paragraph({
        children: [
          new TextRun({ text: `${label}  `, bold: true, color: "AA0000", size: 22 }),
          new TextRun({ text: qr.question, bold: true, size: 22 }),
        ],
        spacing: { before: 280, after: 120 },
      })
    );

    for (const r of qr.responses) {
      children.push(
        new Paragraph({
          children: [
            new TextRun({ text: `${r.speaker_name}`, bold: true }),
            new TextRun({ text: `  (turn ${r.turn_number})`, color: "888888", size: 18 }),
          ],
          spacing: { before: 120, after: 40 },
          indent: { left: 360 },
        })
      );
      children.push(
        new Paragraph({
          text: r.content,
          spacing: { before: 0, after: 80 },
          indent: { left: 360 },
        })
      );

      if (includeNonVerbals) {
        const reactions = qr.nonverbal_reactions.filter(
          (nv) => nv.response_turn === r.turn_number
        );
        if (reactions.length > 0) {
          const reactionText = reactions
            .map((nv) => `[${nv.participant_name}: ${nv.reaction}]`)
            .join("  ");
          children.push(
            new Paragraph({
              children: [new TextRun({ text: reactionText, italics: true, color: "666666", size: 18 })],
              spacing: { before: 0, after: 80 },
              indent: { left: 360 },
            })
          );
        }
      }
    }
  });

  // Stance analysis (if present)
  if (result.stance_matrix_dump) {
    children.push(
      new Paragraph({
        text: "Stance Analysis",
        heading: HeadingLevel.HEADING_1,
        spacing: { before: 360, after: 120 },
      })
    );

    for (const [question, participants] of Object.entries(result.stance_matrix_dump)) {
      children.push(
        new Paragraph({
          text: question,
          heading: HeadingLevel.HEADING_2,
          spacing: { before: 240, after: 80 },
        })
      );

      const participantMap = participants as Record<string, StanceEntry>;
      const dataRows = Object.entries(participantMap ?? {}).map(([name, s]) =>
        new TableRow({
          children: [
            docxCell(name, true),
            docxCell(s.position ?? "—"),
            docxCell(s.sentiment_score != null ? s.sentiment_score.toFixed(1) : "—"),
            docxCell(String(s.contribution_count ?? 0)),
            docxCell((s.key_points ?? []).join("; ") || "—"),
          ],
        })
      );

      children.push(
        new Table({
          width: { size: 100, type: WidthType.PERCENTAGE },
          rows: [
            new TableRow({
              tableHeader: true,
              children: ["Participant", "Position", "Sentiment", "Turns", "Key Points"].map(
                (h) => docxCell(h, true)
              ),
            }),
            ...dataRows,
          ],
        })
      );
    }
  }

  const doc = new Document({
    sections: [{ children }],
  });

  const blob = await Packer.toBlob(doc);
  triggerDownload(blob, safeFilename(sessionName, "docx"));
}

function docxCell(text: string, bold = false): TableCell {
  return new TableCell({
    children: [
      new Paragraph({
        children: [new TextRun({ text, bold, size: 18 })],
      }),
    ],
  });
}
