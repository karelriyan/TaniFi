// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title FarmerRegistry - Soulbound Identity Token for TaniFi Farmers
/// @notice Non-transferable NFT representing farmer identity and reputation
/// @dev Implements Soulbound Token (SBT) pattern - tokens cannot be transferred after minting

contract FarmerRegistry {
    // ============ Structs ============

    struct FarmerProfile {
        uint256 tokenId;
        address walletAddress;
        bytes32 phoneHash;          // Hash of phone number for privacy
        uint256 reputationScore;    // On-chain reputation (starts at 100)
        bool isKYCVerified;
        uint256 registeredAt;
        uint256 completedProjects;
        uint256 failedProjects;
        string metadataURI;         // IPFS hash with additional data
    }

    // ============ State Variables ============

    string public constant name = "TaniFi Farmer Identity";
    string public constant symbol = "TANI-ID";

    address public owner;
    address public taniVault;       // TaniVault contract address for reputation updates

    mapping(address => bool) public kycAdmins;

    // Token storage
    uint256 public totalSupply;
    mapping(uint256 => address) public ownerOf;
    mapping(address => uint256) public tokenOfOwner;  // One token per farmer
    mapping(uint256 => FarmerProfile) public profiles;

    // Phone hash to wallet mapping (prevent duplicate registrations)
    mapping(bytes32 => address) public phoneHashToWallet;

    // Reputation constants
    uint256 public constant INITIAL_REPUTATION = 100;
    uint256 public constant MAX_REPUTATION = 1000;
    uint256 public constant PROJECT_SUCCESS_BONUS = 10;
    uint256 public constant PROJECT_FAILURE_PENALTY = 20;

    // ============ Events ============

    event Transfer(address indexed from, address indexed to, uint256 indexed tokenId);
    event FarmerRegistered(
        uint256 indexed tokenId,
        address indexed farmer,
        bytes32 phoneHash
    );
    event FarmerVerified(address indexed farmer, uint256 tokenId);
    event ReputationUpdated(
        address indexed farmer,
        uint256 oldScore,
        uint256 newScore,
        string reason
    );
    event MetadataUpdated(uint256 indexed tokenId, string newURI);
    event KYCAdminAdded(address indexed admin);
    event KYCAdminRemoved(address indexed admin);

    // ============ Modifiers ============

    modifier onlyOwner() {
        require(msg.sender == owner, "FarmerRegistry: not owner");
        _;
    }

    modifier onlyKYCAdmin() {
        require(kycAdmins[msg.sender] || msg.sender == owner, "FarmerRegistry: not KYC admin");
        _;
    }

    modifier onlyTaniVault() {
        require(msg.sender == taniVault, "FarmerRegistry: not TaniVault");
        _;
    }

    // ============ Constructor ============

    constructor() {
        owner = msg.sender;
        kycAdmins[msg.sender] = true;
    }

    // ============ Registration Functions ============

    /// @notice Register a new farmer (mints SBT)
    /// @param _farmer Wallet address of the farmer
    /// @param _phoneHash Hash of farmer's phone number
    /// @param _metadataURI IPFS URI with additional farmer data
    function registerFarmer(
        address _farmer,
        bytes32 _phoneHash,
        string calldata _metadataURI
    ) external onlyKYCAdmin returns (uint256) {
        require(_farmer != address(0), "FarmerRegistry: invalid farmer");
        require(tokenOfOwner[_farmer] == 0, "FarmerRegistry: already registered");
        require(phoneHashToWallet[_phoneHash] == address(0), "FarmerRegistry: phone already used");

        totalSupply++;
        uint256 tokenId = totalSupply;

        ownerOf[tokenId] = _farmer;
        tokenOfOwner[_farmer] = tokenId;
        phoneHashToWallet[_phoneHash] = _farmer;

        profiles[tokenId] = FarmerProfile({
            tokenId: tokenId,
            walletAddress: _farmer,
            phoneHash: _phoneHash,
            reputationScore: INITIAL_REPUTATION,
            isKYCVerified: false,
            registeredAt: block.timestamp,
            completedProjects: 0,
            failedProjects: 0,
            metadataURI: _metadataURI
        });

        emit Transfer(address(0), _farmer, tokenId);
        emit FarmerRegistered(tokenId, _farmer, _phoneHash);

        return tokenId;
    }

    /// @notice Verify farmer's KYC status
    /// @param _farmer Address of the farmer to verify
    function verifyFarmer(address _farmer) external onlyKYCAdmin {
        uint256 tokenId = tokenOfOwner[_farmer];
        require(tokenId != 0, "FarmerRegistry: farmer not registered");
        require(!profiles[tokenId].isKYCVerified, "FarmerRegistry: already verified");

        profiles[tokenId].isKYCVerified = true;

        emit FarmerVerified(_farmer, tokenId);
    }

    /// @notice Revoke farmer's KYC verification
    /// @param _farmer Address of the farmer
    function revokeVerification(address _farmer) external onlyKYCAdmin {
        uint256 tokenId = tokenOfOwner[_farmer];
        require(tokenId != 0, "FarmerRegistry: farmer not registered");

        profiles[tokenId].isKYCVerified = false;
    }

    // ============ Reputation Functions ============

    /// @notice Update reputation after project completion (called by TaniVault)
    /// @param _farmer Address of the farmer
    /// @param _success Whether the project was successful
    function recordProjectCompletion(address _farmer, bool _success) external onlyTaniVault {
        uint256 tokenId = tokenOfOwner[_farmer];
        require(tokenId != 0, "FarmerRegistry: farmer not registered");

        FarmerProfile storage profile = profiles[tokenId];
        uint256 oldScore = profile.reputationScore;

        if (_success) {
            profile.completedProjects++;
            // Increase reputation (capped at MAX)
            uint256 newScore = oldScore + PROJECT_SUCCESS_BONUS;
            profile.reputationScore = newScore > MAX_REPUTATION ? MAX_REPUTATION : newScore;
            emit ReputationUpdated(_farmer, oldScore, profile.reputationScore, "PROJECT_SUCCESS");
        } else {
            profile.failedProjects++;
            // Decrease reputation (floor at 0)
            if (oldScore > PROJECT_FAILURE_PENALTY) {
                profile.reputationScore = oldScore - PROJECT_FAILURE_PENALTY;
            } else {
                profile.reputationScore = 0;
            }
            emit ReputationUpdated(_farmer, oldScore, profile.reputationScore, "PROJECT_FAILURE");
        }
    }

    /// @notice Manual reputation adjustment (admin only, for special cases)
    /// @param _farmer Address of the farmer
    /// @param _newScore New reputation score
    /// @param _reason Reason for adjustment
    function adjustReputation(
        address _farmer,
        uint256 _newScore,
        string calldata _reason
    ) external onlyOwner {
        uint256 tokenId = tokenOfOwner[_farmer];
        require(tokenId != 0, "FarmerRegistry: farmer not registered");
        require(_newScore <= MAX_REPUTATION, "FarmerRegistry: score too high");

        uint256 oldScore = profiles[tokenId].reputationScore;
        profiles[tokenId].reputationScore = _newScore;

        emit ReputationUpdated(_farmer, oldScore, _newScore, _reason);
    }

    // ============ Metadata Functions ============

    /// @notice Update farmer's metadata URI
    /// @param _farmer Address of the farmer
    /// @param _newURI New IPFS URI
    function updateMetadata(address _farmer, string calldata _newURI) external onlyKYCAdmin {
        uint256 tokenId = tokenOfOwner[_farmer];
        require(tokenId != 0, "FarmerRegistry: farmer not registered");

        profiles[tokenId].metadataURI = _newURI;

        emit MetadataUpdated(tokenId, _newURI);
    }

    /// @notice Get token metadata URI (ERC721 compatible)
    /// @param _tokenId Token ID
    function tokenURI(uint256 _tokenId) external view returns (string memory) {
        require(ownerOf[_tokenId] != address(0), "FarmerRegistry: token not found");
        return profiles[_tokenId].metadataURI;
    }

    // ============ View Functions ============

    /// @notice Get full farmer profile
    /// @param _farmer Address of the farmer
    function getFarmerProfile(address _farmer) external view returns (FarmerProfile memory) {
        uint256 tokenId = tokenOfOwner[_farmer];
        require(tokenId != 0, "FarmerRegistry: farmer not registered");
        return profiles[tokenId];
    }

    /// @notice Get farmer's reputation score
    /// @param _farmer Address of the farmer
    function getReputation(address _farmer) external view returns (uint256) {
        uint256 tokenId = tokenOfOwner[_farmer];
        if (tokenId == 0) return 0;
        return profiles[tokenId].reputationScore;
    }

    /// @notice Check if farmer is registered
    /// @param _farmer Address to check
    function isRegistered(address _farmer) external view returns (bool) {
        return tokenOfOwner[_farmer] != 0;
    }

    /// @notice Check if farmer is KYC verified
    /// @param _farmer Address to check
    function isVerified(address _farmer) external view returns (bool) {
        uint256 tokenId = tokenOfOwner[_farmer];
        if (tokenId == 0) return false;
        return profiles[tokenId].isKYCVerified;
    }

    /// @notice Get balance of address (always 0 or 1 for SBT)
    /// @param _owner Address to check
    function balanceOf(address _owner) external view returns (uint256) {
        return tokenOfOwner[_owner] != 0 ? 1 : 0;
    }

    // ============ Soulbound Transfer Block ============

    /// @notice Transfer function - BLOCKED for Soulbound tokens
    /// @dev Always reverts - tokens are non-transferable
    function transferFrom(address, address, uint256) external pure {
        revert("FarmerRegistry: Soulbound tokens are non-transferable");
    }

    /// @notice Safe transfer function - BLOCKED for Soulbound tokens
    function safeTransferFrom(address, address, uint256) external pure {
        revert("FarmerRegistry: Soulbound tokens are non-transferable");
    }

    /// @notice Safe transfer function with data - BLOCKED for Soulbound tokens
    function safeTransferFrom(address, address, uint256, bytes calldata) external pure {
        revert("FarmerRegistry: Soulbound tokens are non-transferable");
    }

    /// @notice Approve function - BLOCKED for Soulbound tokens
    function approve(address, uint256) external pure {
        revert("FarmerRegistry: Soulbound tokens are non-transferable");
    }

    /// @notice Set approval for all - BLOCKED for Soulbound tokens
    function setApprovalForAll(address, bool) external pure {
        revert("FarmerRegistry: Soulbound tokens are non-transferable");
    }

    /// @notice Get approved address - Always returns zero address for SBT
    function getApproved(uint256) external pure returns (address) {
        return address(0);
    }

    /// @notice Check if approved for all - Always returns false for SBT
    function isApprovedForAll(address, address) external pure returns (bool) {
        return false;
    }

    // ============ Admin Functions ============

    function setTaniVault(address _taniVault) external onlyOwner {
        require(_taniVault != address(0), "FarmerRegistry: invalid TaniVault");
        taniVault = _taniVault;
    }

    function addKYCAdmin(address _admin) external onlyOwner {
        require(_admin != address(0), "FarmerRegistry: invalid admin");
        kycAdmins[_admin] = true;
        emit KYCAdminAdded(_admin);
    }

    function removeKYCAdmin(address _admin) external onlyOwner {
        kycAdmins[_admin] = false;
        emit KYCAdminRemoved(_admin);
    }

    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "FarmerRegistry: invalid owner");
        owner = newOwner;
    }

    // ============ ERC165 Support ============

    function supportsInterface(bytes4 interfaceId) external pure returns (bool) {
        return
            interfaceId == 0x01ffc9a7 || // ERC165
            interfaceId == 0x80ac58cd || // ERC721
            interfaceId == 0x5b5e139f;   // ERC721Metadata
    }
}
