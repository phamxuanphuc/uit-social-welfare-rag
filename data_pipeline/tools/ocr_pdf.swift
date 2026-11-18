// OCR PDF scan bằng Vision (macOS 13+). Build: swiftc -O -o ocr_pdf data_pipeline/tools/ocr_pdf.swift
// Chạy: ./ocr_pdf file.pdf [số_trang_tối_đa] > file.ocr.txt   (in "<<<PAGE n>>>" trước mỗi trang)
import Foundation
import PDFKit
import Vision
let args = CommandLine.arguments
guard args.count >= 2, let doc = PDFDocument(url: URL(fileURLWithPath: args[1])) else { fputs("usage: ocr file.pdf [maxPages]\n", stderr); exit(2) }
let maxPages = args.count > 2 ? Int(args[2])! : doc.pageCount
let req = VNRecognizeTextRequest()
req.recognitionLevel = .accurate
req.usesLanguageCorrection = true
req.recognitionLanguages = ["vi-VN"]
for i in 0..<min(maxPages, doc.pageCount) {
    guard let page = doc.page(at: i) else { continue }
    let box = page.bounds(for: .mediaBox)
    let scale: CGFloat = 3.0
    let img = page.thumbnail(of: CGSize(width: box.width*scale, height: box.height*scale), for: .mediaBox)
    var rect = CGRect(origin: .zero, size: img.size)
    guard let cg = img.cgImage(forProposedRect: &rect, context: nil, hints: nil) else { continue }
    try VNImageRequestHandler(cgImage: cg, options: [:]).perform([req])
    let lines = (req.results ?? []).compactMap { $0.topCandidates(1).first?.string }
    print("<<<PAGE \(i+1)>>>")
    print(lines.joined(separator: "\n"))
}
