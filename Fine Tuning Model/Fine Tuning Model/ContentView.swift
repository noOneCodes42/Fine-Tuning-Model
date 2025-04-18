import SwiftUI
import UniformTypeIdentifiers
import MarkdownUI
// Make sure to disable sandbox (It should be disabled but double check), As well as got System Settings -> Privacy & Security -> Click on "Full Disk Access" and then upload the app into it.
// To upload the app right click the icon on the macOS navigation corresponding to the app and click Show in Finder then drag it into the "Full Disk Access" Area.
// Only HuggingFace Base Models
enum DropAreaError: Error, LocalizedError {
    case fileImportFailed

    var errorDescription: String? {
        switch self {
        case .fileImportFailed:
            return "File import failed."
        }
    }
}

struct ContentView: View {
    @State private var droppedFileName: String? = nil
    @State private var jsonlContent: String = ""
    @State private var modelName: String = ""
    @State private var errorMessage: String? = nil
    @State private var showValidationError: Bool = false
    @FocusState private var isFocused: Bool
    @State private var progress: Double = 0
    @State private var isFineTuning = false
    @State private var fineTuningOutput: String = ""
    @State private var fineTuningStatus: String = "Waiting to start fine-tuning..."
    @State private var scrollToBottomID = UUID()

    @State private var selectedView: String? = "Home" // Track selected sidebar item

    var body: some View {
        NavigationSplitView {
            List(selection: $selectedView) {
                NavigationLink(value: "Home") {
                    Text("Fine Tune Model")
                }
                NavigationLink(value: "Chat") {
                    Text("Chat with Model")
                }
            }
            .navigationTitle("Sidebar")
        } detail: {
            switch selectedView {
            case "Chat":
                ChatView()
            default:
                VStack(spacing: 24) {
                    Text("Type in your Hugging Face base model.")
                        .font(.title2)
                        .frame(maxWidth: .infinity, alignment: .leading)

                    VStack(alignment: .leading, spacing: 4) {
                        TextField("Enter the url for model", text: $modelName)
                            .focused($isFocused)
                            .padding(10)
                            .background(
                                RoundedRectangle(cornerRadius: 8)
                                    .stroke(
                                        showValidationError && modelName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                                        ? Color.red : Color.gray.opacity(0.5),
                                        lineWidth: 1.5
                                    )
                            )
                            .onSubmit { isFocused = false }
                        if showValidationError && modelName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            Text("Required").foregroundColor(.red).font(.caption)
                        }
                    }

                    Text("Drop your .jsonl file below")
                        .font(.title2)
                        .frame(maxWidth: .infinity, alignment: .leading)

                    DropAreaView(
                        highlightRed: showValidationError && droppedFileName == nil
                    ) { result in
                        switch result {
                        case .success(let (fileURL, content)):
                            droppedFileName = fileURL.path
                            jsonlContent = content
                            errorMessage = nil
                            isFocused = false
                        case .failure(let error):
                            droppedFileName = nil
                            jsonlContent = ""
                            errorMessage = error.localizedDescription
                        }
                    }

                    if let error = errorMessage {
                        Text("❌ \(error)")
                            .foregroundColor(.red)
                            .font(.caption)
                    } else if let filePath = droppedFileName {
                        Text("✅ Loaded: \(URL(fileURLWithPath: filePath).lastPathComponent)")
                            .font(.subheadline)
                            .foregroundColor(.green)
                    }

                    if showValidationError && (modelName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || droppedFileName == nil) {
                        Text("*Please finish indicated fields")
                            .foregroundColor(.red)
                            .font(.headline)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.top, 8)
                    }

                    if isFineTuning {
                        VStack(alignment: .leading) {
                            Text(fineTuningStatus)
                                .font(.subheadline)
                                .foregroundColor(.gray)

                            ProgressView(value: progress, total: 1)
                                .progressViewStyle(LinearProgressViewStyle())
                                .padding(.vertical, 4)

                            ScrollView {
                                ScrollViewReader { proxy in
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text(fineTuningOutput)
                                            .font(.system(.body, design: .monospaced))
                                            .frame(maxWidth: .infinity, alignment: .leading)

                                        Color.clear
                                            .frame(height: 1)
                                            .id(scrollToBottomID) // Used as the scroll target
                                    }
                                    .padding(8)
                                    .onChange(of: fineTuningOutput) { _ in
                                        withAnimation {
                                            proxy.scrollTo(scrollToBottomID, anchor: .bottom)
                                        }
                                    }
                                }
                            }
                            .frame(height: 72) // Roughly 3 lines of monospaced body text
                            .background(Color.black.opacity(0.05))
                            .cornerRadius(10)
                        }
                        .padding(.top, 8)
                    }

                    Button(action: {
                        guard !modelName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, let droppedFileName else {
                            showValidationError = true
                            return
                        }

                        showValidationError = false
                        isFineTuning = true
                        progress = 0
                        fineTuningOutput = ""
                        fineTuningStatus = "Starting fine-tuning..."
                        runFineTuning(model: modelName, jsonlPath: droppedFileName)
                    }) {
                        Text("Fine Tune Model")
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.accentColor)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                    }
                    .disabled(isFineTuning)
                    .padding(.top, 12)
                }
                .padding(32)
            }
        }
    }

    func runFineTuning(model: String, jsonlPath: String) {
        let process = Process()
        let pipe = Pipe()
        let fileHandle = pipe.fileHandleForReading

        let scriptPath = "/path/to/the/Finetuining.py" // Change to where the fine tuning script from the gihub is on the device
        let pythonPath = "/path/to/python3" // do which python or which python3 and copy and paste the file path

        process.executableURL = URL(fileURLWithPath: pythonPath)
        process.arguments = [scriptPath, "--model", model, "--data", jsonlPath]
        process.standardOutput = pipe
        process.standardError = pipe

        fileHandle.readabilityHandler = { handle in
            let data = handle.availableData
            guard let output = String(data: data, encoding: .utf8) else { return }

            DispatchQueue.main.async {
                fineTuningOutput += output

                let lines = output.components(separatedBy: .newlines)
                for line in lines {
                    if line.contains("__PROGRESS__") {
                        if let percent = line.components(separatedBy: ":").last,
                           let value = Double(percent.trimmingCharacters(in: .whitespacesAndNewlines)) {
                            progress = value / 100.0
                        }
                    }
                    if line.contains("downloading model") {
                        fineTuningStatus = "Downloading model..."
                        progress = 0.2  // Example to show download progress if available
                    }
                    if line.contains("preparing data") {
                        fineTuningStatus = "Preparing data..."
                        progress = 0.4  // Example
                    }
                    if line.contains("training model") {
                        fineTuningStatus = "Training the model..."
                        progress = 0.7  // Example
                    }
                    if line.contains("saving model") {
                        fineTuningStatus = "Saving the model..."
                        progress = 0.9  // Example
                    }
                    if line.contains("epoch") {
                        fineTuningStatus = "Fine-tuning in progress..."
                    }
                    if line.contains("🎉 Fine-tuning complete") {
                        fineTuningStatus = "Fine-tuning complete!"
                        isFineTuning = false
                        progress = 1.0
                    }
                }
            }
        }

        do {
            try process.run()
        } catch {
            DispatchQueue.main.async {
                isFineTuning = false
                errorMessage = "Failed to start fine-tuning: \(error.localizedDescription)"
            }
        }
    }
}
struct ChatView: View {
    @State private var messageText: String = ""
    @State private var chatMessages: [ChatMessage] = []
    @State private var isGeneratingResponse: Bool = false
    @State private var generatedResponse: String = ""
    @State private var modelFolderPath: String? = nil
    @State private var previousModelFolderPath: String? = nil
    @State private var isModelLoaded: Bool = false
    @State private var loadError: String? = nil
    @State private var isDragOver: Bool = false
    @AppStorage("chatHistory") private var chatHistoryData: Data?

    var body: some View {
        VStack {
            if modelFolderPath == nil || !isModelLoaded {
                // Drag and drop area if model is not loaded
                VStack {
                    Text("Drag and drop your model folder here")
                        .padding()
                        .foregroundColor(isDragOver ? .green : .blue)
                        .frame(maxWidth: .infinity, maxHeight: 200)
                        .background(isDragOver ? Color.gray.opacity(0.3) : Color.gray.opacity(0.1))
                        .cornerRadius(10)
                        .onDrop(of: [.fileURL], isTargeted: $isDragOver) { providers in
                            if let item = providers.first {
                                item.loadObject(ofClass: URL.self) { url, error in
                                    if let url = url, url.hasDirectoryPath {
                                        self.modelFolderPath = url.path
                                        loadModel(from: url.path) // Load model
                                    }
                                }
                            }
                            return true
                        }
                }
                .padding()
            } else {
                // Chat UI after model is loaded
                VStack {
                    ScrollView {
                        VStack(alignment: .leading) {
                            ForEach(chatMessages) { message in
                                ChatMessageView(message: message)
                            }
                            if isGeneratingResponse {
                                Text("Generating response...")
                                    .foregroundColor(.gray)
                                    .padding()
                            } else if !generatedResponse.isEmpty {
                                ChatMessageView(message: ChatMessage(text: generatedResponse, isUser: false))
                            }
                        }
                        .padding()
                    }

                    HStack {
                        TextField("Enter your message", text: $messageText)
                            .padding()
                            .background(Color(.systemGray))
                            .cornerRadius(8)

                        Button("Send") {
                            sendMessage()
                        }
                        .padding()
                        .disabled(isGeneratingResponse || messageText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !isModelLoaded)
                    }
                    .padding()
                }
                
                .overlay(
                    Button("Reset Chat") {
                        resetChat()
                    }
                    .foregroundColor(.red)
                    .background(Color.white.opacity(0.7))
                    .cornerRadius(8)
                    .padding(16),
                    alignment: .topLeading
                )
            }
        }
        .navigationTitle("Chat")
        .onAppear {
            loadChatHistory()
        }

    }

    func sendMessage() {
        guard isModelLoaded else { return } // Don't send messages if model is not loaded

        let userMessage = ChatMessage(text: messageText, isUser: true)
        chatMessages.append(userMessage)
        messageText = ""
        isGeneratingResponse = true

        // Call Python script to get the model's response
        runChatBot(userMessage: userMessage.text) { response in
            DispatchQueue.main.async {
                let modelMessage = ChatMessage(text: response, isUser: false)
                chatMessages.append(modelMessage)
                isGeneratingResponse = false
                saveChatHistory()
            }
        }
        
    }

    func runChatBot(userMessage: String, completion: @escaping (String) -> Void) {
        guard let modelFolderPath = modelFolderPath else {
            completion("Error: Model path not found.")
            return
        }

        let process = Process()
        let pipe = Pipe()
        let fileHandle = pipe.fileHandleForReading

        let scriptPath = "/path/to/the/ChatBot.py" // Adjust path depending on where you clones this repo
        let pythonPath = "/path/to/python3" // do which python or which python3 and copy and paste the file path
        process.executableURL = URL(fileURLWithPath: pythonPath)
        process.arguments = [scriptPath, "--message", userMessage, "--modelPath", modelFolderPath] // Pass the model path
        process.standardOutput = pipe
        process.standardError = pipe

        var outputString = ""

        fileHandle.readabilityHandler = { handle in
            let data = handle.availableData
            if let output = String(data: data, encoding: .utf8) {
                outputString += output
            }
        }

        process.terminationHandler = { _ in
            DispatchQueue.main.async {
                completion(outputString)
            }
        }

        do {
            try process.run()
        } catch {
            DispatchQueue.main.async {
                completion("Error: Could not generate response.")
            }
        }
    }

    func loadModel(from path: String) {
        // Check if the model folder path is different from the previous one
        if modelFolderPath != previousModelFolderPath {
            // Clear chat history if the model path has changed
            chatMessages = []  // Clear previous chat history
            saveChatHistory()  // Clear the saved chat history in case of path change
        }

        isModelLoaded = false // Initially set to false

        // Here, you can add additional model loading checks if needed (e.g., check if the folder is valid).
        // For now, we will just simulate model loading.
        if FileManager.default.fileExists(atPath: path) {
            isModelLoaded = true
            previousModelFolderPath = path // Update previous model path
            loadChatHistory() // Reload chat history if model is loaded
        } else {
            isModelLoaded = false
            chatMessages = [] // Clear chat if model isn't loaded properly
            loadError = "Error: Could not load the model from the provided folder."
        }
    }

    func loadChatHistory() {
        guard let data = chatHistoryData, isModelLoaded else { return }
        if let history = try? JSONDecoder().decode([ChatMessage].self, from: data) {
            chatMessages = history
        }
    }

    func saveChatHistory() {
        if isModelLoaded, let data = try? JSONEncoder().encode(chatMessages) {
            chatHistoryData = data
        }
    }

    func resetChat() {
        // Clear chat history and reset the state
        chatMessages = []
        saveChatHistory()
    }
}






struct DropAreaView: View {
    var highlightRed: Bool
    var onDrop: (Result<(URL, String), DropAreaError>) -> Void
    @State private var animate = false
    @State private var showFileImporter = false

    var body: some View {
        RoundedRectangle(cornerRadius: 12)
            .strokeBorder(
                highlightRed ? Color.red : (animate ? Color.green : Color.blue),
                style: StrokeStyle(lineWidth: 3, dash: [6])
            )
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color.clear)
                    .shadow(color: animate ? Color.green.opacity(0.5) : Color.clear, radius: 10)
            )
            .frame(height: 150)
            .overlay(
                Text("Drag or Click to Select .jsonl File")
                    .foregroundColor(highlightRed ? .red : (animate ? .green : .blue))
                    .bold()
            )
            .animation(.easeInOut(duration: 0.4), value: animate)
            .onDrop(of: [UTType.fileURL], isTargeted: nil) { providers in
                handleDrop(providers: providers)
            }
            .onTapGesture {
                showFileImporter = true
            }
            .fileImporter(
                isPresented: $showFileImporter,
                allowedContentTypes: [.init(filenameExtension: "jsonl")!],
                allowsMultipleSelection: false
            ) { result in
                switch result {
                case .success(let urls):
                    if let url = urls.first {
                        animate = true
                        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                            animate = false
                        }
                        let content = (try? String(contentsOf: url)) ?? ""
                        onDrop(.success((url, content)))
                    }
                case .failure(_):
                    onDrop(.failure(.fileImportFailed))
                }
            }
    }

    private func handleDrop(providers: [NSItemProvider]) -> Bool {
        for provider in providers {
            provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { item, _ in
                DispatchQueue.main.async {
                    guard let data = item as? Data,
                          let url = URL(dataRepresentation: data, relativeTo: nil),
                          url.pathExtension.lowercased() == "jsonl" else {
                        onDrop(.failure(.fileImportFailed))
                        return
                    }
                    animate = true
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                        animate = false
                    }
                    let content = (try? String(contentsOf: url)) ?? ""
                    onDrop(.success((url, content)))
                }
            }
        }
        return true
    }
}

struct ChatMessage: Identifiable, Codable {
    let id = UUID()
    let text: String
    let isUser: Bool
}

struct ChatMessageView: View {
    let message: ChatMessage

    var body: some View {
        HStack {
            if message.isUser {
                Spacer()
            }
            Text(message.text)
                .padding()
                .background(message.isUser ? Color.blue : Color.gray)
                .foregroundColor(.white)
                .cornerRadius(10)
                .frame(maxWidth: .infinity, alignment: message.isUser ? .trailing : .leading)
        }
        .padding(message.isUser ? .leading : .trailing, 50)
    }
}
